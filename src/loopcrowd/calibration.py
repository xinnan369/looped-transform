"""Frozen per-checkpoint count calibration on an explicitly separate fit split."""

import hashlib
import json
from fractions import Fraction

import numpy as np

from .metrics import NAP_THRESHOLDS, count_metrics, image_matches, pooled_point_nap

COUNT_CUTOFF = 0.30
SCORE_FLOOR = 0.05
FLOOR_DIAGNOSTICS = (0.10, 0.20)
GRID_POINTS = tuple(range(5, 96))
CUTOFF_GRID = np.asarray(GRID_POINTS, dtype=np.float64) / 100
DENSITY_GROUPS = ("low", "mid", "high")
FIT_SEED = 42
FIT_PER_GROUP = 20


def json_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def density_group(record, boundaries):
    density = len(record.points) * 1_000_000 / (record.image_width * record.image_height)
    return DENSITY_GROUPS[int(density > boundaries[0]) + int(density > boundaries[1])]


def choose_fit_records(records, boundaries, per_group=FIT_PER_GROUP):
    buckets = {group: [] for group in DENSITY_GROUPS}
    for record in records:
        if record.split == "train":
            key = json_digest(["LC-004B", FIT_SEED, record.dataset, record.image_id])
            buckets[density_group(record, boundaries)].append((key, record))
    manifest, selected = {}, []
    for group, candidates in buckets.items():
        if len(candidates) < per_group:
            raise ValueError(f"fit density bucket {group} has too few original training images")
        ranked = sorted(candidates, key=lambda item: (item[0], item[1].key))[:per_group]
        manifest[group] = [{"image_id": record.image_id, "selection_sha256": key} for key, record in ranked]
        selected.extend(record for _, record in ranked)
    return manifest, selected


def strict_counts(scores, cutoffs):
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    return len(ordered) - np.searchsorted(ordered, cutoffs, side="right")


def fit_cutoff(absolute_error_totals):
    if len(absolute_error_totals) != len(GRID_POINTS):
        raise ValueError("fit errors must cover the frozen cutoff grid")
    index = min(range(len(GRID_POINTS)),
                key=lambda i: (absolute_error_totals[i], abs(GRID_POINTS[i] - 30), -GRID_POINTS[i]))
    return float(CUTOFF_GRID[index]), index


def candidate_digest(predictions):
    """Hash the unchanged coordinates, original scores, source tiles and ownership."""
    sha = hashlib.sha256(bytes.fromhex(json_digest(predictions.tiles)))
    for array, dtype in (
        (predictions.candidates, "<f8"), (predictions.tile_ids, "<i8"), (predictions.owned, "u1"),
    ):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        sha.update(bytes.fromhex(json_digest([canonical.dtype.str, canonical.shape])))
        sha.update(canonical.tobytes())
    return sha.hexdigest()


def remap_scores(scores, fitted_cutoff):
    """Express a fitted count cutoff at 0.30 using a monotonic odds multiplier."""
    scores = np.asarray(scores, dtype=np.float64)
    scale = (COUNT_CUTOFF / (1 - COUNT_CUTOFF)) * ((1 - fitted_cutoff) / fitted_cutoff)
    numerator = scale * scores
    return numerator / (1 - scores + numerator)


def count_match_metrics(points, candidates, cutoffs):
    ranked, matches = image_matches(points, candidates, (0.50,))
    scores = candidates[ranked, 2]
    counts = np.searchsorted(-scores, -np.asarray(cutoffs), side="left")
    true_positives = np.r_[0, np.cumsum(matches[0] >= 0)][counts]
    return [
        {
            "predicted": int(count), "count_error": int(count) - len(points),
            "abs_count_error": abs(int(count) - len(points)),
            "local_tp": int(tp), "local_fp": int(count - tp), "local_fn": int(len(points) - tp),
        }
        for count, tp in zip(counts, true_positives)
    ]


def rank_metrics(points, predictions):
    ranked, matches = image_matches(points, predictions.candidates, NAP_THRESHOLDS)
    return predictions.candidates[ranked, 2], matches >= 0


def fixed_pool_ap(ranked, reference_heads, fitted_cutoff):
    scores, assignments = zip(*ranked)
    before = np.concatenate(scores)
    after = remap_scores(before, fitted_cutoff)
    original_order = np.argsort(-before, kind="stable")
    remapped_order = np.argsort(-after, kind="stable")
    changed_ranks = int(np.count_nonzero(original_order != remapped_order))
    source_boundaries = before[original_order][1:] != before[original_order][:-1]
    mapped_in_source_order = after[original_order]
    new_ties = int(np.count_nonzero(
        source_boundaries & (mapped_in_source_order[1:] == mapped_in_source_order[:-1])
    ))
    original = pooled_point_nap(scores, assignments, reference_heads)
    remapped = pooled_point_nap([remap_scores(vector, fitted_cutoff) for vector in scores],
                               assignments, reference_heads) if changed_ranks == 0 else None
    return {
        "original": original,
        "count_calibrated_fixed_pool": original,
        "remap_check": remapped,
        "remap_changed_rank_positions": changed_ranks,
        "remap_new_ties_between_distinct_scores": new_ties,
        "original_tie_positions": int(len(source_boundaries) - source_boundaries.sum()),
    }


def summarize_counts(rows, mode):
    gt = [row["gt"] for row in rows]
    predicted = [row[mode]["predicted"] for row in rows]
    counts = {key: sum(row[mode][key] for row in rows) for key in ("local_tp", "local_fp", "local_fn")}
    return {
        **count_metrics(predicted, gt), "images": len(rows), "gt": sum(gt), "predicted": sum(predicted),
        **counts, "local_precision": counts["local_tp"] / sum(predicted) if sum(predicted) else None,
        "local_recall": counts["local_tp"] / sum(gt) if sum(gt) else None,
    }


def gap_summary(raw_gaps, corrected_gaps, rank_invariant):
    if len(raw_gaps) != 3 or len(corrected_gaps) != 3:
        raise ValueError("the frozen comparison uses exactly three seeds")
    mean_raw = sum(raw_gaps) / len(raw_gaps)
    mean_corrected = sum(corrected_gaps) / len(corrected_gaps)
    per_seed = [float((raw - corrected) / raw) if raw > 0 else None
                for raw, corrected in zip(raw_gaps, corrected_gaps)]
    mean_closure = float((mean_raw - mean_corrected) / mean_raw) if mean_raw > 0 else None
    improved = sum(raw > 0 and corrected < raw for raw, corrected in zip(raw_gaps, corrected_gaps))
    return {
        "raw_mean_gap": float(mean_raw), "corrected_mean_gap": float(mean_corrected),
        "mean_gap_closure": mean_closure, "per_seed_closure": per_seed,
        "seeds_with_narrower_positive_raw_gap": improved, "rank_invariant": rank_invariant,
        "pass": mean_closure is not None and mean_closure >= 0.50 and improved >= 2 and rank_invariant,
    }


def exact_mae_fraction(sum_absolute_errors, images):
    return Fraction(int(sum_absolute_errors), images)
