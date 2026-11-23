"""Count metrics and the documented local implementation of point nAP."""

import numpy as np
from scipy.spatial import cKDTree

NAP_THRESHOLDS = tuple(index / 20 for index in range(1, 11))


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


def point_nap(samples, thresholds=NAP_THRESHOLDS, neighbors=3):
    """Accept (ground-truth Nx2, predicted Mx3) pairs in native image coordinates."""
    thresholds = tuple(thresholds)
    if not thresholds or min(thresholds) <= 0 or neighbors < 1:
        raise ValueError("nAP thresholds and number of neighbors must be positive")
    ranked = []
    sizes = []
    for image, (points, predictions) in enumerate(samples):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        predictions = np.asarray(predictions, dtype=float).reshape(-1, 3)
        sizes.append(len(points))
        candidates = [[] for _ in predictions]
        if len(points) and len(predictions):
            tree = cKDTree(points)
            normalizer = np.ones(len(points))
            if len(points) > 1:
                distances, _ = tree.query(points, k=1 + min(neighbors, len(points) - 1))
                normalizer = distances[:, 1:].mean(axis=1).clip(min=1e-12)
            predicted_tree = cKDTree(predictions[:, :2])
            neighborhoods = predicted_tree.query_ball_point(points, max(thresholds) * normalizer)
            for gt_id, ids in enumerate(neighborhoods):
                if len(ids):
                    ratios = np.linalg.norm(points[gt_id] - predictions[ids, :2], axis=1) / normalizer[gt_id]
                    for predicted_id, ratio in zip(ids, ratios):
                        candidates[predicted_id].append((ratio, gt_id))
            for nearest in candidates:
                nearest.sort()
        ranked.extend((score, image, index, candidates[index]) for index, (_, _, score) in enumerate(predictions))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

    average_precision = {}
    head_count = sum(sizes)
    for threshold in thresholds:
        matched = set()
        matches = []
        for _, image, _, candidates in ranked:
            pair = next(((image, index) for distance, index in candidates
                         if distance < threshold and (image, index) not in matched), None)
            matches.append(pair is not None)
            if pair is not None:
                matched.add(pair)
        average_precision[f"{threshold:g}"] = _ap(matches, head_count)
    valid = [value for value in average_precision.values() if value is not None]
    return {"nap": float(np.mean(valid)) if valid else None, "ap_by_delta": average_precision}
