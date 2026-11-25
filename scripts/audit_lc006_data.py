"""Verify the frozen ShanghaiTech B train and internal validation data locally."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from loopcrowd.audit import _grid_histogram, _grid_summary
from loopcrowd.calibration import density_group, json_digest
from loopcrowd.crops import crop_origin, crop_points
from loopcrowd.datasets import _by_stem, _record, _shanghai_points
from loopcrowd.experiment import file_sha, write_json
from loopcrowd.manifest import read_manifest
from loopcrowd.model import pretrained_path
from scripts.run_lc003 import WEIGHT_SHA, load_json

MANIFEST_B_SHA = "de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482"
REJECTIONS_B_SHA = "37b7cef9d3b43591453b90c085ab7528db4038ae6c605fa8d014def54d5692ca"
DATA_CONFIG = Path("configs/shanghaitech_b.json")
HISTORICAL_AUDIT = Path("reports/LC-001/shanghaitech_b_audit.json")


def evidence(path):
    return {"path": str(path.resolve()), "sha256": file_sha(path), "bytes": path.stat().st_size}


def grid_report(histogram, heads):
    values = _grid_summary(histogram, heads)
    return {key: value for key, value in values.items() if key != "overflow"} | {
        "heads": heads, "stride": 4, "slots": 2, "overflow": values["overflow"]["2"],
    }


def audit_data(manifest, root, rejection_file):
    manifest_info, rejections_info, weights = map(evidence, (manifest, rejection_file, pretrained_path()))
    if (manifest_info["sha256"], rejections_info["sha256"], weights["sha256"]) != (
        MANIFEST_B_SHA, REJECTIONS_B_SHA, WEIGHT_SHA,
    ):
        raise ValueError("frozen B manifest, rejections or pretrained cache digest differs")
    records = read_manifest(manifest, splits=("train", "val"))
    config = load_json(DATA_CONFIG)
    if config["coordinate_base"] != 0 or config["validation"] != {"fraction": 0.10, "seed": 42}:
        raise ValueError("B manifest protocol differs from the historical train split")
    official_train = root / config["split_dirs"]["train"]
    image_files = _by_stem(official_train / "images", config["image_suffix"])
    annotation_files = _by_stem(official_train / "ground_truth", ".mat")
    ids = {f"IMG_{i}" for i in range(1, 401)}
    if set(image_files) != ids or set(annotation_files) != {f"GT_{stem}" for stem in ids}:
        raise ValueError("original B official-train image or annotation IDs differ")
    ranked = sorted((f"train/{stem}" for stem in ids), key=lambda item: (hashlib.sha256(
        f"42:{item}".encode()).digest(), item))
    validation_ids = set(ranked[:40])
    if {row.image_id for row in records} != set(ranked) or len(records) != len(ranked):
        raise ValueError("original train400 IDs differ from train360/val40 manifest membership")
    groups = {"train": [], "val": []}
    raw_heads = Counter()
    rejected = []
    verified_files = []
    for row in records:
        split = "val" if row.image_id in validation_ids else "train"
        stem = row.image_id.removeprefix("train/")
        image, annotation = image_files[stem], annotation_files[f"GT_{stem}"]
        if row.split != split or row.dataset != "shanghaitech_b" or Path(row.image_path).resolve() != image.resolve():
            raise ValueError(f"{row.image_id}: frozen split, dataset or image path mismatch")
        with Image.open(image) as decoded:
            decoded.load()
        points = _shanghai_points(annotation)
        replay = _record(config, image, row.image_id, "train", points, rejected)
        if (replay.image_width, replay.image_height, replay.points) != (row.image_width, row.image_height, row.points):
            raise ValueError(f"{row.image_id}: B label replay differs from the original manifest")
        raw_heads[split] += len(points)
        groups[split].append(row)
        verified_files.append([row.image_id, file_sha(image), file_sha(annotation)])
    by_point = lambda row: (row["image_id"], row["point_index"])
    archived = [row for row in load_json(rejection_file) if row["official_split"] == "train"]
    if sorted(rejected, key=by_point) != sorted(archived, key=by_point):
        raise ValueError("label cleaning differs from the frozen B official-train rejection ledger")
    density = [len(row.points) * 1_000_000 / (row.image_width * row.image_height) for row in groups["train"]]
    boundaries = np.quantile(density, [1 / 3, 2 / 3]).tolist()
    historical = load_json(HISTORICAL_AUDIT)
    splits = {}
    for split, rows in groups.items():
        baseline = next(group for group in historical["groups"]
                        if group["split"] == split and group["mode"] == "full-image")
        heads = sum(len(row.points) for row in rows)
        expected = {"train": (360, 44856), "val": (40, 4295)}[split]
        if (len(rows), heads) != expected or expected != (baseline["image_count"], baseline["head_count"]):
            raise ValueError(f"historical B cleaned head or image count mismatch in {split}")
        full, crop = Counter(), Counter()
        sampled = 0
        for row in rows:
            points = np.asarray(row.points, dtype=float).reshape(-1, 2)
            width, height = row.image_width, row.image_height
            full.update(_grid_histogram(points, width, height, 4))
            left, top = crop_origin(points, width, height, 512, "random", 42, f"{row.dataset}:{row.image_id}", 1)
            retained = crop_points(points, left, top, 512)
            sampled += len(retained)
            crop.update(_grid_histogram(retained, min(width, 512), min(height, 512), 4))
        full_summary = grid_report(full, heads)
        if full_summary["overflow"] != baseline["grids"]["4"]["overflow"]["2"]:
            raise ValueError(f"historic full-image stride4/K2 capacity differs in {split}")
        splits[split] = {
            "images": len(rows), "raw_heads": raw_heads[split], "cleaned_heads": heads,
            "rejected_out_of_bounds_heads": raw_heads[split] - heads,
            "split_ids_sha256": json_digest([row.image_id for row in rows]),
            "density_tertile_image_counts": dict(Counter(density_group(row, boundaries) for row in rows)),
            "full_image_capacity": full_summary, "random_crop_probe_capacity": grid_report(crop, sampled),
        }
    return {
        "execution": "passed", "manifest": manifest_info, "historical_capacity_audit": evidence(HISTORICAL_AUDIT),
        "historical_label_rejections": rejections_info, "dataset_config": evidence(DATA_CONFIG),
        "pretrained_weight": weights, "dataset_root": str(root.resolve()),
        "train_image_annotation_pair_digests_sha256": json_digest(verified_files),
        "official_train_file_pairs_verified": len(verified_files), "official_test_images_opened": 0,
        "coordinate_base": 0, "coordinate_policy": "reject only finite out-of-frame points and match ledger exactly",
        "split_policy": "first 40 of 400 train IDs sorted by sha256('42:' + image_id), then ID",
        "train_density_tertiles_heads_per_megapixel": boundaries,
        "crop_capacity_probe": {"size": 512, "policy": "random", "seed": 42, "epoch": 1},
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_data(args.manifest, args.root, args.rejections)
    write_json(args.output, report)
    for split, row in report["splits"].items():
        print(split, row["images"], row["raw_heads"], row["cleaned_heads"], row["rejected_out_of_bounds_heads"])


if __name__ == "__main__":
    main()
