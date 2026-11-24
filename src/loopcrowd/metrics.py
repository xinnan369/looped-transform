"""Counting, deterministic local nAP and validation-only score sweeps."""

import numpy as np
from scipy.spatial import cKDTree

NAP_THRESHOLDS = tuple(index / 20 for index in range(1, 11))
SCORE_THRESHOLDS = tuple(index / 20 for index in range(1, 20))


def count_metrics(predicted, reference):
    predicted, reference = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if predicted.ndim != 1 or predicted.shape != reference.shape or not len(reference):
        raise ValueError("count metrics require aligned nonempty image counts")
    error = predicted - reference
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def _ap(matches, number_of_heads):
    if not number_of_heads:
        return None
    tp = np.cumsum(matches)
    recall = np.r_[0.0, tp / number_of_heads, 1.0]
    precision = np.r_[0.0, tp / np.arange(1, len(tp) + 1), 0.0]
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * envelope[1:]))


def image_matches(points, predictions, thresholds, neighbors=3):
    """Return original prediction IDs and GT matches, both in stable score order."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 3)
    ranked = np.argsort(-predictions[:, 2], kind="stable")
    ranked_xy = predictions[ranked, :2]
    matched = np.full((len(thresholds), len(ranked)), -1, dtype=int)
    if not len(points) or not len(ranked_xy):
        return ranked, matched

    normalizer = np.ones(len(points))
    if len(points) > 1:
        distances, _ = cKDTree(points).query(points, k=1 + min(neighbors, len(points) - 1))
        normalizer = distances[:, 1:].mean(axis=1).clip(min=1e-12)
    # The KD-tree is built in score order. Its indices address ranks, not
    # original input rows, and match columns use that same rank coordinate.
    neighborhoods = cKDTree(ranked_xy).query_ball_point(points, max(thresholds) * normalizer)
    candidates_by_rank = {}
    for gt_id, ranks in enumerate(neighborhoods):
        if len(ranks):
            ratios = np.linalg.norm(points[gt_id] - ranked_xy[ranks], axis=1) / normalizer[gt_id]
            for rank, ratio in zip(ranks, ratios):
                candidates_by_rank.setdefault(rank, []).append((ratio, gt_id))
    ordered = [(rank, sorted(candidates_by_rank[rank])) for rank in sorted(candidates_by_rank)]
    for row, threshold in enumerate(thresholds):
        used = set()
        for rank, nearby in ordered:
            for distance, gt_id in nearby:
                if distance >= threshold:
                    break
                if gt_id not in used:
                    matched[row, rank] = gt_id
                    used.add(gt_id)
                    break
    return ranked, matched


def point_nap(samples, thresholds=NAP_THRESHOLDS, neighbors=3):
    """Accept (ground-truth Nx2, predicted Mx3) pairs in native image coordinates."""
    thresholds = tuple(thresholds)
    if not thresholds or min(thresholds) <= 0 or neighbors < 1:
        raise ValueError("nAP thresholds and number of neighbors must be positive")
    scores, assignments, head_count = [], [], 0
    for points, predictions in samples:
        predictions = np.asarray(predictions, dtype=float).reshape(-1, 3)
        ranked, matched = image_matches(points, predictions, thresholds, neighbors)
        scores.append(predictions[ranked, 2])
        assignments.append(matched >= 0)
        head_count += len(points)
    order = np.argsort(-np.concatenate(scores), kind="stable") if scores else []
    ranked_matches = np.concatenate(assignments, axis=1)[:, order] if scores else np.zeros((len(thresholds), 0))
    average_precision = {f"{threshold:g}": _ap(matches, head_count)
                         for threshold, matches in zip(thresholds, ranked_matches)}
    valid = [value for value in average_precision.values() if value is not None]
    return {"nap": float(np.mean(valid)) if valid else None, "ap_by_delta": average_precision}


def score_threshold_sweep(samples, thresholds=SCORE_THRESHOLDS):
    """Use count-side ownership candidates and local one-to-one matching at δ=0.50."""
    thresholds = np.asarray(thresholds)
    gt, predicted_counts, true_positives = [], [], np.zeros(len(thresholds), dtype=int)
    for points, predictions in samples:
        predictions = np.asarray(predictions, dtype=float).reshape(-1, 3)
        ranked, matched = image_matches(points, predictions, (0.5,))
        # The score boundary and TP prefix refer to the same rank positions.
        scores = predictions[ranked, 2]
        per_threshold = np.searchsorted(-scores, -thresholds, side="left")
        predicted_counts.append(per_threshold)
        true_positives += np.r_[0, np.cumsum(matched[0] >= 0)][per_threshold]
        gt.append(len(points))
    if not gt:
        raise ValueError("score sweep requires labeled validation images")
    gt_total = sum(gt)
    rows = []
    for threshold, counts, tp in zip(thresholds, np.asarray(predicted_counts).T, true_positives):
        total = int(counts.sum())
        rows.append({
            "threshold": float(threshold), **count_metrics(counts, gt),
            "predicted_heads": total, "gt_heads": gt_total,
            "local_true_positives_delta_0_5": int(tp),
            "local_precision": float(tp / total) if total else None,
            "local_recall": float(tp / gt_total) if gt_total else None,
        })
    return rows
