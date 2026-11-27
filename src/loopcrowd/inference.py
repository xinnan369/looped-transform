"""Full-frame tiled inference with source-aware, distinct count and AP views."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .data import image_tile
from .model.refinement import midpoint_uncertainty


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def tile_axis(length, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("tile size must be positive and overlap must be in [0, size)")
    last = max(0, length - size)
    origins = list(range(0, last + 1, size - overlap))
    if origins[-1] != last:
        origins.append(last)
    boundaries = [0] + [(left + size + right) / 2 for left, right in zip(origins, origins[1:])] + [length]
    return list(zip(origins, boundaries, boundaries[1:]))


def decode(prediction, cutoff, stride=4, *, with_ids=False):
    scores = prediction["logits"][0].float().sigmoid()
    slots, rows, cols = torch.where(scores > cutoff)
    offsets = prediction["offsets"][0, slots, :, rows, cols].float()
    x = (cols + offsets[:, 0]) * stride
    y = (rows + offsets[:, 1]) * stride
    values = torch.stack((x, y, scores[slots, rows, cols]), dim=1).cpu().numpy()
    if with_ids:
        return values, torch.stack((rows, cols, slots), dim=1).cpu().numpy()
    return values


@dataclass(frozen=True)
class TilePredictions:
    """Every in-frame AP candidate with its source tile and ownership flag."""

    candidates: np.ndarray
    tile_ids: np.ndarray
    owned: np.ndarray
    tiles: list[dict]
    candidate_ids: np.ndarray | None = None

    def count_candidates(self):
        return self.candidates[self.owned]

    def diagnostics(self, count_threshold):
        high_score = self.candidates[:, 2] > count_threshold
        counted = self.owned & high_score
        points = self.candidates[counted, :2]
        ids = self.tile_ids[counted]
        pairs = cKDTree(points).query_pairs(2.0, output_type="ndarray")
        first, second = ids[pairs[:, 0]], ids[pairs[:, 1]]
        # A possibly ambiguous pair must originate from different tiles whose
        # valid image regions overlap. It is recorded and both points are kept.
        boxes = np.asarray([tile["image_xyxy"] for tile in self.tiles])
        overlap = (np.minimum(boxes[first, 2:], boxes[second, 2:])
                   > np.maximum(boxes[first, :2], boxes[second, :2])).all(axis=1)
        return {
            "ap_candidates": len(self.candidates),
            "count_excluded_by_ownership": int((high_score & ~self.owned).sum()),
            "ambiguous_owned_pairs_within_2px": int(((first != second) & overlap).sum()),
            "reconciled_pairs": 0,
        }

    def export(self):
        result = {
            "tiles": self.tiles,
            "columns": ["x", "y", "score", "tile_id", "owned"],
            "candidates": [row + [tile_id, owned] for row, tile_id, owned in
                           zip(self.candidates.tolist(), self.tile_ids.tolist(), self.owned.tolist())],
        }
        if self.candidate_ids is not None:
            result.update(candidate_id_columns=["tile", "row", "col", "slot"],
                          candidate_ids=self.candidate_ids.tolist())
        return result


@torch.inference_mode()
def predict_image(model, image, config, device, *, loop_count=None, trace=None, require_finite=False,
                  export_identity=False):
    return _predict_tiles(model, image, config, device, loop_count=loop_count, trace=trace,
                          require_finite=require_finite, export_identity=export_identity)["final"]


@torch.inference_mode()
def predict_image_stages(model, image, config, device, *, require_finite=False):
    return _predict_tiles(model, image, config, device, staged=True, require_finite=require_finite)


@torch.inference_mode()
def predict_image_depth(model, image, config, device, *, diagnostic=True, export_identity=False,
                        midpoint_maps=None, refinement_gains=None):
    norms = []
    predictions = _predict_tiles(model, image, config, device, depth_norms=norms, require_finite=True,
                                 depth_diagnostic=diagnostic, export_identity=export_identity, midpoint_maps=midpoint_maps,
                                 refinement_gains=refinement_gains)
    return predictions, np.stack(norms)


def _predict_tiles(model, image, config, device, *, loop_count=None, trace=None, staged=False, require_finite=False,
                   depth_norms=None, depth_diagnostic=True, export_identity=False, midpoint_maps=None,
                   refinement_gains=None):
    size, overlap = config["tile_size"], config["tile_overlap"]
    stages = ("early", "final") if staged else ("final",)
    if depth_norms is not None:
        selected = model.depth_readout_stages(depth_diagnostic)
        stages = tuple(selected.values())
    by_stage = {stage: ([], [], []) for stage in stages}
    candidate_keys = {stage: [] for stage in stages} if export_identity else {}
    tiles = []
    model_args = {}
    if loop_count is not None:
        model_args["loop_count"] = loop_count
    if trace is not None:
        model_args["trace"] = trace
    for top, y_min, y_max in tile_axis(image.height, size, overlap):
        for left, x_min, x_max in tile_axis(image.width, size, overlap):
            tile, width, height = image_tile(image, left, top, size)
            tile_id = len(tiles)
            tiles.append({
                "tile_id": tile_id, "image_xyxy": [left, top, left + width, top + height],
                "ownership_xyxy": [x_min, y_min, x_max, y_max],
            })
            with autocast(device, config["precision"]):
                batch = tile[None].to(device)
                refined = getattr(model, "refinement", None) is not None
                point_valid = None
                if refined or midpoint_maps is not None:
                    axes = torch.arange((size + model.stride - 1) // model.stride, device=device) * model.stride
                    point_valid = ((axes[:, None] < height) & (axes[None, :] < width))[None, None]
                refinement_args = {"point_valid": point_valid} if refined else {}
                if refined and refinement_gains is not None:
                    refinement_args["refinement_trace"] = lambda gain: refinement_gains.append(gain[0].cpu().numpy())
                if depth_norms is not None:
                    sums = []

                    def trace_state(index, before, after):
                        if index in selected:
                            previous, current = before.float(), after.float()
                            sums.append(torch.stack((previous.square().mean(), (current - previous).square().mean())))

                    outputs = model.forward_depth_stages(batch, diagnostic=depth_diagnostic, state_trace=trace_state,
                                                         **refinement_args)
                    depth_norms.append(torch.stack(sums).cpu().numpy())
                    if not np.isfinite(depth_norms[-1]).all():
                        raise ValueError("nonfinite depth state update")
                else:
                    outputs = (model.forward_stages(batch) if staged else
                               {"final": model(batch, **model_args, **refinement_args)})
                if midpoint_maps is not None:
                    scores, valid = midpoint_uncertainty(outputs["early"]["logits"], point_valid)
                    midpoint_maps.append((scores[0].cpu().numpy(), valid[0].cpu().numpy()))
            if require_finite:
                finite = [torch.isfinite(output[key]).all() for output in outputs.values() for key in ("logits", "offsets")]
                if not torch.stack(finite).all():
                    raise ValueError("nonfinite tiled logits or offsets")
            for stage, output in outputs.items():
                predictions, ids, ownership = by_stage[stage]
                decoded = decode(output, config["localization_min_score"], model.stride, with_ids=export_identity)
                if export_identity:
                    decoded, cells = decoded
                decoded = decoded.astype(float)
                valid = ((decoded[:, :2] >= 0) & (decoded[:, :2] < [width, height])).all(axis=1)
                points = decoded[valid]
                points[:, :2] += [left, top]
                owned = ((points[:, :2] >= [x_min, y_min]) & (points[:, :2] < [x_max, y_max])).all(axis=1)
                predictions.append(points)
                source = np.full(len(points), tile_id, dtype=int)
                ids.append(source)
                ownership.append(owned)
                if export_identity:
                    candidate_keys[stage].append(np.column_stack((source, cells[valid])))
    return {stage: TilePredictions(*(np.concatenate(values) for values in arrays), tiles,
                                   np.concatenate(candidate_keys[stage]) if export_identity else None)
            for stage, arrays in by_stage.items()}
