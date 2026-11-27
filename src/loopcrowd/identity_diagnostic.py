"""Opt-in identity records from the unchanged point matcher."""

import numpy as np

from .metrics import NAP_THRESHOLDS, image_matches, point_spacing


def matched_identities(points, prediction, *, threshold=None):
    """Keep candidate row IDs, scores, and GT assignment rows in stable rank order."""
    if prediction.candidate_ids is None:
        raise ValueError("candidate identities were not retained")
    selected = np.arange(len(prediction.candidates))
    deltas = NAP_THRESHOLDS
    if threshold is not None:
        selected = selected[prediction.owned & (prediction.candidates[:, 2] > threshold)]
        deltas = (.50,)
    rank, gt = image_matches(points, prediction.candidates[selected], deltas)
    selected = selected[rank]
    return {"candidate_rows": selected, "candidate_ids": prediction.candidate_ids[selected],
            "candidates": prediction.candidates[selected], "gt_ids": gt, "deltas": deltas}


def count_identity_details(points, prediction, threshold):
    """Attribute count-side false positives and retain every GT ID in the denominator."""
    record = matched_identities(points, prediction, threshold=threshold)
    assigned = record["gt_ids"][0]
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    matched = {int(gt): rank for rank, gt in enumerate(assigned) if gt >= 0}
    unmatched = sorted(set(range(len(points))) - matched.keys())
    fp = np.flatnonzero(assigned < 0)
    labels = []
    radii = .50 * point_spacing(points)
    for rank in fp:
        inside = np.flatnonzero(np.linalg.norm(points - record["candidates"][rank, :2], axis=1) < radii)
        if any(gt in matched for gt in inside):
            labels.append("duplicate_near_matched_gt")
        elif len(inside):
            labels.append("unassigned_near_gt")
        else:
            labels.append("background_outside_gt_radius")
    return record | {"matched_ranks": matched, "unmatched_gt_ids": unmatched,
                     "false_positive_ranks": fp, "false_positive_kind": labels}


def gt_transitions(before, after):
    """Match GT rows across stages independently from candidate ID changes."""
    left, right = before["matched_ranks"], after["matched_ranks"]
    common = sorted(left.keys() & right.keys())
    changes = []
    for gt in common:
        a, b = left[gt], right[gt]
        first, last = before["candidates"][a], after["candidates"][b]
        changes.append({"gt_id": gt, "before_candidate_id": before["candidate_ids"][a].tolist(),
                        "after_candidate_id": after["candidate_ids"][b].tolist(),
                        "displacement": float(np.linalg.norm(last[:2] - first[:2])),
                        "score_change": float(last[2] - first[2]), "rank_change": b - a})
    return {"recovered": sorted(right.keys() - left.keys()), "lost": sorted(left.keys() - right.keys()),
            "unmatched_both": sorted(set(before["unmatched_gt_ids"]) & set(after["unmatched_gt_ids"])),
            "matched_both": changes}


def uncertainty_at_points(points, prediction, midpoint_maps):
    """Sample the one ownership tile and its aligned stride-8 cell at each position."""
    coords = np.asarray(points, dtype=float).reshape(-1, 2)
    values = np.full(len(coords), np.nan, dtype=float)
    for tile, (score, support) in zip(prediction.tiles, midpoint_maps, strict=True):
        left, top, right, bottom = tile["ownership_xyxy"]
        inside = ((coords[:, :2] >= [left, top]) & (coords[:, :2] < [right, bottom])).all(axis=1)
        if np.isfinite(values[inside]).any():
            raise ValueError("ownership tiles overlap for a diagnostic position")
        x, y = tile["image_xyxy"][:2]
        col, row = np.floor((coords[inside] - [x, y]) / 8).astype(int).T
        if not support[row, col].all():
            raise ValueError("a diagnostic position falls outside the valid token grid")
        values[inside] = score[row, col]
    if not np.isfinite(values).all():
        raise ValueError("a diagnostic position has no valid midpoint score")
    return values


def confidence_association(points, midpoint, final, midpoint_maps, threshold=.30):
    before = count_identity_details(points, midpoint, threshold)
    after = count_identity_details(points, final, threshold)
    gt_scores = uncertainty_at_points(points, midpoint, midpoint_maps)
    candidate_uncertainty = uncertainty_at_points(before["candidates"][:, :2], midpoint, midpoint_maps)
    p = before["candidates"][:, 2].clip(1e-12, 1 - 1e-12)
    return {
        "gt_id": np.arange(len(gt_scores)), "gt_midpoint_uncertainty": gt_scores,
        "gt_matched_midpoint": np.isin(np.arange(len(gt_scores)), list(before["matched_ranks"])),
        "gt_matched_final": np.isin(np.arange(len(gt_scores)), list(after["matched_ranks"])),
        "midpoint_candidate_ids": before["candidate_ids"],
        "midpoint_candidate_uncertainty": candidate_uncertainty,
        "midpoint_candidate_gt_ids": before["gt_ids"][0],
        "midpoint_score_margin": before["candidates"][:, 2] - threshold,
        "midpoint_candidate_entropy": -(p * np.log(p) + (1 - p) * np.log(1 - p)),
        "midpoint_false_positive_kind": before["false_positive_kind"],
        "midpoint_false_positive_ranks": before["false_positive_ranks"],
        "gt_transitions": gt_transitions(before, after),
    }
