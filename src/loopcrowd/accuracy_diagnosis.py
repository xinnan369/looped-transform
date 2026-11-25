"""Frozen validation diagnostics used by LC-007A."""

import numpy as np
from scipy.spatial import cKDTree

from .audit import _grid_histogram
from .calibration import COUNT_CUTOFF, DENSITY_GROUPS, candidate_digest
from .metrics import NAP_THRESHOLDS, count_metrics, image_matches, point_spacing, pooled_point_nap

SCORE_EDGES = tuple(index / 20 for index in range(1, 21))
MATCH_DELTA = 0.50
BOOTSTRAP_SEED = 7007
BOOTSTRAP_SAMPLES = 2000


def density(record):
    return len(record.points) * 1_000_000 / (record.image_width * record.image_height)


def density_bin(value, boundaries):
    return DENSITY_GROUPS[np.searchsorted(boundaries, value, side="right")]


def distribution(values):
    if not len(values):
        return dict.fromkeys(("mean", "p10", "median", "p90", "max"))
    low, median, high = np.quantile(values, (0.1, 0.5, 0.9))
    return dict(zip(("mean", "p10", "median", "p90", "max"),
                    map(float, (np.mean(values), low, median, high, np.max(values)))))


def near_ownership_seam(points, tiles, width, height):
    boxes = np.array([tile["ownership_xyxy"] for tile in tiles])
    near = np.zeros(len(points), dtype=bool)
    for axis, length in enumerate((width, height)):
        edges = np.unique(boxes[:, (axis, axis + 2)])
        internal = edges[(edges > 0) & (edges < length)]
        near |= (np.abs(points[:, axis, None] - internal) <= 4).any(axis=1)
    return near


def analyze_image(record, predictions, boundaries):
    points = np.asarray(record.points, dtype=float).reshape(-1, 2)
    dimensions = (record.image_width, record.image_height)
    raw = predictions.candidates
    owned = predictions.count_candidates()
    counted = owned[owned[:, 2] > COUNT_CUTOFF]
    digest = candidate_digest(predictions)

    order, matched = image_matches(points, counted, (MATCH_DELTA,))
    accepted = matched[0] >= 0
    gt_ids = matched[0, accepted]
    distances = np.linalg.norm(counted[order[accepted], :2] - points[gt_ids], axis=1)
    spacing = point_spacing(points)
    nearest = cKDTree(points).query(points, k=2)[0][:, 1] if len(points) > 1 else []
    gt_seam = near_ownership_seam(points, predictions.tiles, *dimensions)
    count_seam = near_ownership_seam(counted[:, :2], predictions.tiles, *dimensions)
    raw_seam = near_ownership_seam(raw[:, :2], predictions.tiles, *dimensions)
    overflow = sum(max(0, occupancy - 2) * cells
                   for occupancy, cells in _grid_histogram(points, *dimensions, 4).items())

    def covered(candidates):
        if not len(points) or not len(candidates):
            return 0
        distance = cKDTree(candidates[:, :2]).query(points)[0]
        return int((distance / spacing < MATCH_DELTA).sum())

    error = len(counted) - len(points)
    row = {
        "image_id": record.image_id, "width": dimensions[0], "height": dimensions[1],
        "density_heads_per_mpx": density(record), "density_group": density_bin(density(record), boundaries),
        "gt": len(points), "predicted": len(counted), "signed_error": error, "absolute_error": abs(error),
        "tp": len(gt_ids), "fp": len(counted) - len(gt_ids), "fn": len(points) - len(gt_ids),
        "tiles": len(predictions.tiles), "tile_geometry": predictions.tiles,
        "ap_floor_near_seam": int(raw_seam.sum()), "owned_floor": len(owned),
        "owned_floor_near_seam": int(raw_seam[predictions.owned].sum()),
        "count_near_seam": int(count_seam.sum()), "gt_near_seam": int(gt_seam.sum()),
        "fn_near_seam": int(gt_seam.sum() - gt_seam[gt_ids].sum()),
        "fp_near_seam": int(count_seam[order[~accepted]].sum()),
        "covered_raw_floor": covered(raw), "covered_owned_floor": covered(owned),
        "covered_count_mask": covered(counted), "stride4_k2_frame_overflow_proxy": overflow,
        "score_hist_raw": np.histogram(raw[:, 2], SCORE_EDGES)[0].tolist(),
        "score_hist_owned": np.histogram(owned[:, 2], SCORE_EDGES)[0].tolist(),
        "score_hist_counted": np.histogram(counted[:, 2], SCORE_EDGES)[0].tolist(),
        "candidate_sha256": digest,
    } | predictions.diagnostics(COUNT_CUTOFF)
    for name, values in (("nearest_gt_px", nearest), ("matcher_spacing_px", spacing if len(points) > 1 else []),
                         ("matched_px", distances), ("matched_normalized", distances / spacing[gt_ids])):
        row.update({f"{name}_{key}": value for key, value in distribution(values).items()})
    ranks, assignments = image_matches(points, raw, NAP_THRESHOLDS)
    if digest != candidate_digest(predictions):
        raise AssertionError("diagnosis mutated the original candidate pool")
    return row, (raw[ranks, 2], assignments >= 0)


def aggregate(rows, ranked):
    if not rows:
        return {"images": 0}
    gt = [row["gt"] for row in rows]
    scores, assignments = zip(*ranked)
    result = count_metrics([row["predicted"] for row in rows], gt)
    result["local_nap"] = pooled_point_nap(scores, assignments, sum(gt))["nap"]
    return result | {"images": len(rows)} | {
        key: sum(row[key] for row in rows)
        for key in ("gt", "predicted", "tp", "fp", "fn", "tiles", "ap_candidates", "owned_floor",
                    "gt_near_seam", "fn_near_seam", "fp_near_seam", "covered_raw_floor",
                    "covered_owned_floor", "covered_count_mask", "stride4_k2_frame_overflow_proxy")
    }


def paired_rows(target, reference):
    def keys(rows):
        return [(row["image_id"], row["width"], row["height"], row["gt"], row["density_group"]) for row in rows]

    target_keys = keys(target)
    if not target_keys or len({key[0] for key in target_keys}) != len(target) or target_keys != keys(reference):
        raise ValueError("paired runs require the same ordered, unique validation images and labels")
    fields = ("absolute_error", "signed_error", "predicted", "tp", "fp", "fn", "covered_raw_floor",
              "covered_owned_floor", "covered_count_mask", "fn_near_seam", "fp_near_seam")
    return [
        {key: a[key] for key in ("image_id", "density_group", "gt")} | {
            "reference_absolute_error": b["absolute_error"], "target_absolute_error": a["absolute_error"],
        } | {f"delta_{key}": a[key] - b[key] for key in fields}
        for a, b in zip(target, reference)
    ]


def bootstrap_indices(number_of_images):
    if number_of_images < 2:
        raise ValueError("sampling sensitivity requires at least two images")
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, number_of_images, (BOOTSTRAP_SAMPLES, number_of_images), dtype=np.int64)


def sampling_sensitivity(paired_differences, indices):
    differences = np.asarray(paired_differences, dtype=float)
    if differences.ndim != 2 or not len(differences) or differences.shape[1] != indices.shape[1]:
        raise ValueError("expected one aligned vector of image differences per training seed")
    seed_deltas = differences.mean(axis=1)
    by_image = differences.mean(axis=0)
    omitted = (by_image.sum() - by_image) / (len(by_image) - 1)
    bootstrap_means = by_image[indices].mean(axis=1)
    return {
        "full_validation_mean": float(by_image.mean()), "per_seed_full_validation": seed_deltas.tolist(),
        "training_seed_sample_sd": float(seed_deltas.std(ddof=1)) if len(seed_deltas) > 1 else None,
        "leave_one_image_out": omitted.tolist(),
        "bootstrap_p025_median_p975": np.quantile(bootstrap_means, (0.025, 0.5, 0.975)).tolist(),
    }
