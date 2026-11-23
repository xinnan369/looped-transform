#!/usr/bin/env python3
"""Audit random and density-biased crop capacity on train and internal validation."""

import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np

from loopcrowd.audit import CAPACITIES, CROP_SIZE, STRIDES, _grid_histogram, _grid_summary
from loopcrowd.crops import crop_origin, crop_points
from loopcrowd.manifest import read_manifest


def summarize(records, policy, seed):
    rows, total_heads = [], 0
    totals = {stride: Counter() for stride in STRIDES}
    for record in records:
        points = np.asarray(record.points, dtype=float).reshape(-1, 2)
        left, top = crop_origin(points, record.image_width, record.image_height, CROP_SIZE,
                                policy, seed, f"{record.dataset}:{record.image_id}")
        cropped = crop_points(points, left, top, CROP_SIZE)
        total_heads += len(cropped)
        grids = {}
        for stride in STRIDES:
            histogram = _grid_histogram(cropped, CROP_SIZE, CROP_SIZE, stride)
            totals[stride].update(histogram)
            grids[str(stride)] = _grid_summary(histogram, len(cropped))["overflow"]
        rows.append({"image_id": record.image_id, "origin_xy": [left, top],
                     "source_heads": len(points), "crop_heads": len(cropped), "overflow": grids})
    return rows, {
        "images": len(rows), "source_heads": sum(row["source_heads"] for row in rows),
        "crop_heads": total_heads,
        "overflow": {str(stride): _grid_summary(total, total_heads)["overflow"]
                     for stride, total in totals.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    groups = []
    for manifest in args.manifest:
        records = read_manifest(manifest)
        datasets = {record.dataset for record in records}
        if len(datasets) != 1:
            parser.error("each manifest must contain exactly one dataset")
        dataset = next(iter(datasets))
        for split in ("train", "val"):
            selected = [record for record in records if record.split == split]
            if not selected:
                parser.error(f"{manifest}: {split} is empty")
            for seed in (42, 123, 2026):
                for policy in ("random", "density-biased"):
                    rows, summary = summarize(selected, policy, seed)
                    group = {"dataset": dataset, "split": split, "policy": policy, "seed": seed, **summary}
                    if not args.aggregate_only:
                        group["per_image"] = rows
                    groups.append(group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"crop_size": CROP_SIZE, "capacities": CAPACITIES, "groups": groups}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(groups)} audit groups to {args.output}")


if __name__ == "__main__":
    main()
