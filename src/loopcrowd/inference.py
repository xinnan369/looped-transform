"""Full-frame tiled inference with source-aware, distinct count and AP views."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .data import image_tile


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


def decode(prediction, cutoff, stride=4):
    scores = prediction["logits"][0].float().sigmoid()
    slots, rows, cols = torch.where(scores > cutoff)
    offsets = prediction["offsets"][0, slots, :, rows, cols].float()
    x = (cols + offsets[:, 0]) * stride
    y = (rows + offsets[:, 1]) * stride
    return torch.stack((x, y, scores[slots, rows, cols]), dim=1).cpu().numpy()


@dataclass(frozen=True)
class TilePredictions:
    """Every in-frame AP candidate with its source tile and ownership flag."""

    candidates: np.ndarray
    tile_ids: np.ndarray
    owned: np.ndarray
    tiles: list[dict]

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
        return {
            "tiles": self.tiles,
            "columns": ["x", "y", "score", "tile_id", "owned"],
            "candidates": [row + [tile_id, owned] for row, tile_id, owned in
                           zip(self.candidates.tolist(), self.tile_ids.tolist(), self.owned.tolist())],
        }


@torch.inference_mode()
def predict_image(model, image, config, device, *, loop_count=None, trace=None):
    size, overlap = config["tile_size"], config["tile_overlap"]
    predictions, ids, ownership, tiles = [], [], [], []
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
                output = model(tile[None].to(device), **model_args)
            decoded = decode(output, config["localization_min_score"], model.stride).astype(float)
            # Filter the valid side of each partially padded tile before returning native coordinates.
            points = decoded[((decoded[:, :2] >= 0) & (decoded[:, :2] < [width, height])).all(axis=1)]
            points[:, :2] += [left, top]
            owned = ((points[:, :2] >= [x_min, y_min]) & (points[:, :2] < [x_max, y_max])).all(axis=1)
            predictions.append(points)
            ids.append(np.full(len(points), tile_id, dtype=int))
            ownership.append(owned)
    return TilePredictions(np.concatenate(predictions), np.concatenate(ids), np.concatenate(ownership), tiles)
