#!/usr/bin/env python3
"""Verify local matching against a dense oracle and inspect the pinned upstream release."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np

from loopcrowd.metrics import NAP_THRESHOLDS, SCORE_THRESHOLDS, image_matches, point_nap, score_threshold_sweep

UPSTREAM = "5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455"
FIXTURES = {
    "empty": [([], [])],
    "single": [([[3, 4]], [[3, 4, 0.9], [9, 9, 0.1]])],
    "two_close": [([[8, 8], [9, 8]], [[8, 8, 0.8], [9, 8, 0.7]])],
    "duplicate": [([[3, 4], [13, 4]], [[3, 4, 0.9], [3, 4, 0.8], [13, 4, 0.5]])],
    "low_score_first": [([[0, 0]], [[0.1, 0, 0.1], [0, 0, 0.9]])],
    "equal_scores": [([[8, 8]], [[20, 20, 0.7], [8, 8, 0.7]])],
    "equal_scores_across_images": [([], [[0, 0, 0.7]]), ([[0, 0]], [[0, 0, 0.7]])],
    "interleaved_image_scores": [
        ([[0, 0]], [[0, 0, 0.4]]),
        ([[4, 4]], [[4, 4, 0.8], [0, 0, 0.95]]),
    ],
    "competing_gt": [([[0, 0], [2, 0], [30, 0], [60, 0]], [[1, 0, 0.9], [0, 0, 0.5]])],
    "isolated_images": [([[8, 8]], []), ([], [[8, 8, 0.9]])],
    "delta_boundary": [([[0, 0], [10, 0]], [[5, 0, 0.8]])],
}


def dense_oracle(samples, delta):
    normalizers = []
    for gt, _ in samples:
        widths = []
        for index, point in enumerate(gt):
            nearest = sorted(math.dist(point, other) for k, other in enumerate(gt) if index != k)[:3]
            widths.append(max(sum(nearest) / len(nearest), 1e-12) if nearest else 1)
        normalizers.append(widths)
    order = sorted(
        ((-point[2], image, index) for image, (_, predictions) in enumerate(samples)
         for index, point in enumerate(predictions))
    )
    used = set()
    matching = []
    for _, image, index in order:
        gt, predictions = samples[image]
        matching.append(None)
        distances = sorted(
            (math.dist(point, predictions[index][:2]) / normalizers[image][k], k)
            for k, point in enumerate(gt)
        )
        for ratio, k in distances:
            if ratio < delta and (image, k) not in used:
                used.add((image, k))
                matching[-1] = k
                break
    total = sum(len(gt) for gt, _ in samples)
    if not total:
        return order, matching, None
    tp, precision = 0, []
    for index, value in enumerate(matching, 1):
        tp += value is not None
        precision.append(tp / index)
    ap = sum(max(precision[index:]) for index, value in enumerate(matching)
             if value is not None) / total
    return order, matching, ap


def verify_fixture(samples, thresholds):
    metric = point_nap(samples, thresholds)
    local = [image_matches(gt, predictions, thresholds) for gt, predictions in samples]
    local_order = [(image, int(p)) for image, (indices, _) in enumerate(local) for p in indices]
    scores = np.asarray([samples[i][1][p][2] for i, p in local_order], dtype=float)
    local_order = [local_order[index] for index in np.argsort(-scores, kind="stable")]
    rows = []
    for row, delta in enumerate(thresholds):
        order, oracle_gt, oracle_ap = dense_oracle(samples, delta)
        assert local_order == [item[1:] for item in order]
        local_gt = []
        actual = {
            (image, int(index)): int(matches[row, rank])
            for image, (indices, matches) in enumerate(local)
            for rank, index in enumerate(indices)
        }
        for _, image, index in order:
            index = actual[(image, index)]
            local_gt.append(index if index >= 0 else None)
        assert local_gt == oracle_gt
        local_ap = metric["ap_by_delta"][f"{delta:g}"]
        difference = None if oracle_ap is None else local_ap - oracle_ap
        assert (local_ap is None if oracle_ap is None else abs(difference) < 1e-12)
        rows.append({"delta": delta, "order_image_and_prediction": [item[1:] for item in order],
                     "matched_gt_indices": local_gt, "local_ap": local_ap, "dense_oracle_ap": oracle_ap,
                     "local_minus_oracle": difference, "upstream_ap": None, "upstream_delta": None})
    return {"images": [{"gt": gt, "predictions": predictions} for gt, predictions in samples],
            "thresholds": rows, "nap_local_v1": metric["nap"]}


def verify_sweep(samples):
    order, matches, _ = dense_oracle(samples, 0.50)
    gt = sum(len(points) for points, _ in samples)
    rows = []
    for row in score_threshold_sweep(samples, SCORE_THRESHOLDS):
        retained = [match for (negative_score, _, _), match in zip(order, matches)
                    if -negative_score > row["threshold"]]
        tp = sum(match is not None for match in retained)
        assert row["predicted_heads"] == len(retained)
        assert row["local_true_positives_delta_0_5"] == tp
        assert row["local_precision"] == (tp / len(retained) if retained else None)
        assert row["local_recall"] == (tp / gt if gt else None)
        rows.append({**row, "dense_oracle_tp": tp, "local_minus_oracle_tp": 0})
    return rows


def inspect_release(path):
    with tarfile.open(path) as archive:
        members = {m.name.split("/", 1)[1]: m for m in archive.getmembers() if m.isfile()}
        required = ("engine.py", "run_test.py", "models/matcher.py", "README.md")
        assert all(UPSTREAM in members[name].name for name in required)
        text = {}
        for name, member in members.items():
            if name.endswith(".py") or name in required:
                stream = archive.extractfile(member)
                assert stream is not None
                text[name] = stream.read().decode()
    details = {}
    for name in required:
        relevant = [index for index, line in enumerate(text[name].splitlines(), 1)
                    if re.search(r"\bnAP\b|def evaluate|predict_cnt|linear_sum_assignment|np\.sqrt|cv2\.imwrite", line)]
        details[name] = {"sha256": hashlib.sha256(text[name].encode()).hexdigest(),
                         "matching_line_numbers": relevant}
    locations = [name for name, source in text.items()
                 if name.endswith(".py") and re.search(r"\b(?:nap|normalized_average_precision)\b", source, re.I)]
    assert not locations, locations
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for block in iter(lambda: archive.read(1024**2), b""):
            digest.update(block)
    return {
        "commit": UPSTREAM,
        "archive_sha256": digest.hexdigest(),
        "github_tree": f"https://api.github.com/repos/TencentYoutuResearch/CrowdCounting-P2PNet/git/trees/{UPSTREAM}?recursive=1",
        "python_files_inspected": sorted(name for name in text if name.endswith(".py")),
        "evidence": details,
        "status": "unavailable_upstream_evaluator",
        "reason": "The pinned release includes count evaluation, visualization, and a training matcher. It contains the paper nAP table but no executable nAP evaluator.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "metric": "nap_local_v1", "parity": "local_only",
        "upstream": inspect_release(args.upstream_archive),
        "paper": "https://openaccess.thecvf.com/content/ICCV2021/html/Song_Rethinking_Counting_and_Localization_in_Crowds_A_Purely_Point-Based_Framework_ICCV_2021_paper.html",
        "fixtures": {name: {**verify_fixture(samples, NAP_THRESHOLDS), "threshold_sweep": verify_sweep(samples)}
                     for name, samples in FIXTURES.items()},
        "boundary_above_point_five": verify_fixture(FIXTURES["delta_boundary"], (0.50, 0.51)),
        "nwpu_official_f1": "not_ready",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified local ranking, matching, AP and sweep TP prefixes on {len(FIXTURES)} fixtures.")


if __name__ == "__main__":
    main()
