"""Internal-validation diagnostics for fixed loop prefixes."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn.functional import cosine_similarity

from .data import load_rgb
from .inference import predict_image
from .metrics import NAP_THRESHOLDS, count_metrics, image_matches, point_spacing, pooled_point_nap

SCORE_BINS = np.arange(1, 21) / 20
DENSITY_GROUPS = ("low", "mid", "high")


class LoopStatistics:
    """Accumulate per-tile statistics on device, with one transfer per result."""

    fields = (
        "input_rms", "output_rms", "relative_update_l2", "cosine_similarity",
        "spatial_std_input", "spatial_std_output", "channel_std_input", "channel_std_output",
        "attention_residual_rms", "ffn_residual_rms",
    )

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def __call__(self, stage, before, after, attention, feedforward):
        before, after = before.detach().float(), after.detach().float()
        spatial_before, spatial_after = before.flatten(1, 2), after.flatten(1, 2)
        flat_before, flat_after = before.flatten(1), after.flatten(1)
        change = torch.linalg.vector_norm(flat_after - flat_before, dim=1)
        initial = torch.linalg.vector_norm(flat_before, dim=1)
        values = torch.stack((
            flat_before.square().mean(dim=1).sqrt(), flat_after.square().mean(dim=1).sqrt(),
            change / (initial + 1e-12), cosine_similarity(flat_before, flat_after, eps=1e-12),
            spatial_before.std(dim=1, unbiased=False).mean(dim=1),
            spatial_after.std(dim=1, unbiased=False).mean(dim=1),
            spatial_before.std(dim=2, unbiased=False).mean(dim=1),
            spatial_after.std(dim=2, unbiased=False).mean(dim=1),
            attention.detach().float().flatten(1).square().mean(dim=1).sqrt(),
            feedforward.detach().float().flatten(1).square().mean(dim=1).sqrt(),
        )).sum(dim=1)
        self.sums[stage] = self.sums.get(stage, 0) + values
        self.counts[stage] = self.counts.get(stage, 0) + len(before)

    def result(self):
        return {
            str(stage + 1): dict(zip(self.fields, (values / self.counts[stage]).tolist()))
            | {"tiles": self.counts[stage]}
            for stage, values in self.sums.items()
        }


def backbone_bn_buffers(model):
    result = {}
    for group in ("p2", "p3"):
        for name, layer in getattr(model, group).named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                result[f"{group}.{name}"] = {
                    "running_mean": layer.running_mean.detach().cpu().numpy().copy(),
                    "running_var": layer.running_var.detach().cpu().numpy().copy(),
                    "num_batches_tracked": int(layer.num_batches_tracked),
                }
    return result


def describe_bn(buffers, reference=None):
    def describe(values):
        return {
            "q05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)), "rms": float(np.sqrt(np.mean(values**2))),
        }

    result = {}
    for layer, buffers_by_name in buffers.items():
        row = {"num_batches_tracked": buffers_by_name["num_batches_tracked"]}
        for key in ("running_mean", "running_var"):
            values = buffers_by_name[key]
            row[key] = describe(values)
            if reference is not None:
                delta = values - reference[layer][key]
                row[key]["rms_delta_from_pretrained"] = float(np.sqrt(np.mean(delta**2)))
                row[key]["max_abs_delta_from_pretrained"] = float(np.max(np.abs(delta)))
        result[layer] = row
    return result


def _match_counts(points, predictions):
    ranked, ranked_matches = image_matches(points, predictions, (0.50,))
    matches = np.full(len(ranked), -1, dtype=int)
    matches[ranked] = ranked_matches[0]
    matched = matches >= 0
    close = cKDTree(predictions[:, :2]).query_pairs(2.0, output_type="ndarray")
    distinct = matched[close].all(axis=1) & (matches[close[:, 0]] != matches[close[:, 1]])
    radii = 0.5 * point_spacing(points)
    neighborhoods = cKDTree(predictions[:, :2]).query_ball_point(points, radii)
    eligible = np.zeros(len(ranked), dtype=bool)
    gt_with_competition = 0
    for point, radius, nearby in zip(points, radii, neighborhoods):
        indices = np.asarray(nearby, dtype=int)
        inside = indices[np.linalg.norm(predictions[indices, :2] - point, axis=1) < radius]
        eligible[inside] = True
        gt_with_competition += int(len(inside) > 1)
    tp = int(matched.sum())
    return {
        "local_tp": tp, "local_fp": len(predictions) - tp, "local_fn": len(points) - tp,
        "gt_with_multiple_eligible_predictions": gt_with_competition,
        "unmatched_predictions_with_eligible_gt": int((eligible & ~matched).sum()),
        "unmatched_predictions_outside_all_gt_radii": int((~eligible & ~matched).sum()),
        "within_2px_pairs_distinct_matched_gt": int(distinct.sum()),
        "within_2px_pairs_without_two_distinct_matches": int((~distinct).sum()),
    }


def _aggregate(rows, ranked):
    score_vectors, match_matrices = zip(*ranked)
    gt = [row["gt"] for row in rows]
    observed = [row["predicted"] for row in rows]
    fields = rows[0]["local_matches"].keys() | rows[0]["conflicts"].keys()
    return {
        "images": len(rows), "gt": sum(gt), "predicted": sum(observed),
        **count_metrics(observed, gt), "count_signed_bias": (sum(observed) - sum(gt)) / len(rows),
        "local_nap": pooled_point_nap(score_vectors, match_matrices, sum(gt)),
        "score_histogram": np.sum([row["score_histogram"] for row in rows], axis=0).tolist(),
        "local_match_and_conflict_totals": {
            key: sum(row["local_matches"].get(key, row["conflicts"].get(key, 0)) for row in rows)
            for key in sorted(fields)
        },
    }


def evaluate_prefix(model, records, manifest_dir, config, device, boundaries, calls, *, verify_default=False):
    """Evaluate one fixed prefix with the unchanged LC-003 candidate and matching protocol."""
    model.eval()
    rows, ranked = [], []
    stats = LoopStatistics()
    cutoff = config["count_threshold"]
    for record in records:
        points = np.asarray(record.points, dtype=float).reshape(-1, 2)
        density = len(points) * 1e6 / (record.image_width * record.image_height)
        group = DENSITY_GROUPS[int(density > boundaries[0]) + int(density > boundaries[1])]
        with load_rgb(record, manifest_dir) as image:
            preds = predict_image(model, image, config, device, loop_count=calls, trace=stats)
            if verify_default:
                ordinary = predict_image(model, image, config, device)
                for key in ("candidates", "tile_ids", "owned"):
                    if not np.array_equal(getattr(preds, key), getattr(ordinary, key)):
                        raise AssertionError(f"full prefix changes default-path candidates on {record.image_id}")
        scores = preds.candidates[:, 2]
        count_set = preds.count_candidates()
        count_set = count_set[count_set[:, 2] > cutoff]
        order, matches = image_matches(points, preds.candidates, NAP_THRESHOLDS)
        ranked.append((scores[order], matches >= 0))
        rows.append({
            "image_id": record.image_id, "density_heads_per_megapixel": density, "density_group": group,
            "gt": len(points), "predicted": len(count_set), "count_error": len(count_set) - len(points),
            "absolute_count_error": abs(len(count_set) - len(points)),
            "local_nap": pooled_point_nap([scores[order]], [matches >= 0], len(points))["nap"],
            "score_histogram": np.histogram(scores, SCORE_BINS)[0].tolist(),
            "local_matches": _match_counts(points, count_set), "conflicts": preds.diagnostics(cutoff),
        })
    groups = {}
    for group in DENSITY_GROUPS:
        selected = [(row, pairs) for row, pairs in zip(rows, ranked) if row["density_group"] == group]
        groups[group] = _aggregate(*zip(*selected)) if selected else {"images": 0}
    return {
        "calls": calls, "overall": _aggregate(rows, ranked), "density_groups": groups, "per_image": rows,
        "loop_statistics": stats.result(), "full_prefix_bitwise_vs_default": True if verify_default else None,
    }


def compare_prefixes(one_call, two_calls):
    pairs = list(zip(one_call["per_image"], two_calls["per_image"], strict=True))
    assert all(a["image_id"] == b["image_id"] and a["gt"] == b["gt"] for a, b in pairs)
    deltas = [
        {"image_id": b["image_id"], "density_group": b["density_group"],
         "absolute_count_error_delta": b["absolute_count_error"] - a["absolute_count_error"],
         "signed_count_error_delta": b["count_error"] - a["count_error"],
         "local_nap_delta": b["local_nap"] - a["local_nap"] if a["local_nap"] is not None else None}
        for a, b in pairs
    ]

    def summary(rows):
        changes = [row["absolute_count_error_delta"] for row in rows]
        count = len(changes)
        better, worse = sum(delta < 0 for delta in changes), sum(delta > 0 for delta in changes)
        return {"images": count, "improved": better, "degraded": worse, "tied": count - better - worse,
                "improved_fraction": better / count if count else None,
                "degraded_fraction": worse / count if count else None,
                "mae_delta": float(np.mean(changes)) if count else None}

    return {
        "overall": summary(deltas), "density_groups": {
            group: summary([row for row in deltas if row["density_group"] == group]) for group in DENSITY_GROUPS
        },
        "rmse_delta": two_calls["overall"]["rmse"] - one_call["overall"]["rmse"],
        "local_nap_delta": two_calls["overall"]["local_nap"]["nap"] - one_call["overall"]["local_nap"]["nap"],
        "per_image": deltas,
    }
