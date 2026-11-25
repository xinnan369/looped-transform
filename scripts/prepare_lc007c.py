"""Freeze the LC-007C crop plan from train annotations without opening images or checkpoints."""

import argparse
from pathlib import Path
import json

import numpy as np

from loopcrowd.accuracy_diagnosis import density_bin
from loopcrowd.calibration import DENSITY_GROUPS, json_digest
from loopcrowd.crops import crop_origin, crop_points
from loopcrowd.experiment import file_sha, read_config, write_json
from loopcrowd.gradient_diagnosis import TOLERANCES, WEIGHT, ZERO_NORM
from loopcrowd.manifest import read_manifest
from scripts.archive_lc007c import ARCHIVE, FREEZE, OUTPUT, PARENT, POLICIES, SEEDS, SHARED, UNSHARED
from scripts.run_lc007a import read_csv, write_csv

MANIFEST_SHA = "de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482"
WEIGHT_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
TAG = "LC-007C/train-only/crop-probes-v1"
CROP_EPOCH = 40
PER_GROUP = 8
PILOT_CAP = {"cases": 96, "gpu_seconds": 1200, "process_peak_reserved_bytes": 2_147_483_648,
             "project_optimizer_updates": 0, "experiment_download_bytes": 0}


def crop_plan(train, seed):
    if any(record.split != "train" for record in train):
        raise ValueError("probe crop selection accepts only original train records")
    size = 512
    candidates = []
    for record in train:
        points = np.asarray(record.points, dtype=float).reshape(-1, 2)
        left, top = crop_origin(points, record.image_width, record.image_height, size, "random", seed,
                                f"{record.dataset}:{record.image_id}", CROP_EPOCH)
        width, height = min(size, record.image_width - left), min(size, record.image_height - top)
        heads = len(crop_points(points, left, top, size))
        selection = json_digest([TAG, seed, record.image_id, left, top])
        candidates.append({"image_id": record.image_id, "left": left, "top": top,
                           "valid_width": width, "valid_height": height, "crop_gt": heads,
                           "heads_per_mpx": heads * 1_000_000 / (width * height), "selection_sha256": selection})
    boundaries = np.quantile([crop["heads_per_mpx"] for crop in candidates], (1 / 3, 2 / 3)).tolist()
    if not boundaries[0] < boundaries[1]:
        raise ValueError(f"non-distinct crop-density tertiles for seed {seed}")
    batches, totals = [], {}
    for group in DENSITY_GROUPS:
        subset = [crop for crop in candidates if density_bin(crop["heads_per_mpx"], boundaries) == group]
        totals[group] = len(subset)
        if len(subset) < PER_GROUP:
            raise ValueError(f"seed {seed} has {len(subset)} eligible {group} crops; requires {PER_GROUP}")
        selected = sorted(subset, key=lambda crop: (crop["selection_sha256"], crop["image_id"]))[:PER_GROUP]
        for start in range(0, PER_GROUP, 2):
            batch = {"index": len(batches) + 1, "density_group": group, "crops": selected[start:start + 2]}
            batches.append(batch | {"batch_sha256": json_digest(batch)})
    return {"seed": seed, "train_crop_boundaries_heads_per_mpx": boundaries, "eligible_by_group": totals,
            "batches": batches, "batch_sequence_sha256": json_digest(batches)}


def verify_checkpoints():
    matrix = {(int(row["seed"]), row["arm"]): row for row in read_csv(ARCHIVE / "experiment_matrix.csv")}
    evidence = []
    for seed in SEEDS:
        for arm in (SHARED, UNSHARED):
            row = matrix[seed, arm]
            checkpoint = json.loads(row["last_checkpoint"])
            index = read_csv(ARCHIVE / "epochs" / str(seed) / arm / "training.csv")[-1]
            config_path = Path(f"configs/lc006_{arm}.json")
            config = read_config(config_path) | {"seed": seed, "ema_decay": .99}
            if (row["training_code_sha"] != FREEZE or row["manifest_sha256"] != MANIFEST_SHA
                    or row["pretrained_sha256"] != WEIGHT_SHA or index["epoch"] != "40"
                    or index["checkpoint_sha256"] != checkpoint["sha256"]
                    or json_digest(config) != row["config_sha256"]):
                raise ValueError(f"frozen config or E40 provenance differs for {seed}/{arm}")
            local = Path(checkpoint["path"])
            if local.stat().st_size != checkpoint["bytes"] or file_sha(local) != checkpoint["sha256"]:
                raise ValueError(f"local checkpoint byte integrity differs for {seed}/{arm}")
            evidence.append({"seed": seed, "arm": arm, "epoch": 40, "checkpoint": checkpoint,
                             "effective_config_sha256": row["config_sha256"],
                             "source_config": str(config_path), "source_config_sha256": file_sha(config_path),
                             "byte_integrity": "verified_without_deserialization"})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path.home() / "loopcrowd-lc002/shanghaitech_b.jsonl")
    parser.add_argument("--weight", type=Path, default=Path.home() /
                        "loopcrowd-lc002/torch/hub/checkpoints/resnet18-f37072fd.pth")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "execution": "not_run_pending_owner_gpu_approval", "archive_commit": PARENT,
        "checkpoint_training_commit": FREEZE, "proposed_limits": PILOT_CAP, "policies": POLICIES,
        "frozen_scoring": {"count_cutoff_strict": .30, "ap_score_floor_strict": .05},
        "crop_policy": {"size": 512, "mode": "random", "origin_epoch": CROP_EPOCH, "population": "train-only",
                        "tag": TAG, "density_binning": "linear 1/3, 2/3 quantiles, lower-inclusive mid/high",
                        "ordering": "low, mid, high; eight lowest selection SHA per group, consecutive pairs"},
        "fp32_near_zero_norm": ZERO_NORM, "aux_loss_weight": WEIGHT, "tolerances_abs_rel": TOLERANCES,
        "preparation": "local files and training annotations only; no image opens or checkpoint deserialization",
    }
    try:
        if file_sha(args.manifest) != MANIFEST_SHA or file_sha(args.weight) != WEIGHT_SHA:
            raise ValueError("local B manifest or cached initial weight checksum mismatch")
        train = read_manifest(args.manifest, splits=("train",))
        if len(train) != 360 or sum(len(record.points) for record in train) != 44_856:
            raise ValueError("frozen B train population differs")
        result["dataset"] = {"path": str(args.manifest), "sha256": MANIFEST_SHA,
                             "train_images": 360, "cleaned_train_gt": 44_856}
        result["weight_cache"] = {"path": str(args.weight), "sha256": WEIGHT_SHA}
        result["checkpoints"] = verify_checkpoints()
        result["plans"] = [crop_plan(train, seed) for seed in SEEDS]
        result["planned_batch_policy_cases"] = sum(len(row["batches"]) for row in result["plans"]) * 2 * len(POLICIES)
        if result["planned_batch_policy_cases"] != PILOT_CAP["cases"]:
            raise ValueError("proposed case budget differs from the frozen plan")
    except (OSError, ValueError) as error:
        result.update(execution="blocked_local_inputs", reason=str(error))
    write_json(args.output / "probe_manifest.json", result)
    write_csv(args.output / "gradient_diagnostics.csv", [
        {"seed": seed, "arm": arm, "policy": policy, "expected_batches": 12,
         "actual_cases": 0, "status": result["execution"], "cosine": None}
        for seed in SEEDS for arm in (SHARED, UNSHARED) for policy in POLICIES
    ])
    print(json.dumps({"execution": result["execution"], "cases": result.get("planned_batch_policy_cases", 0),
                      "reason": result.get("reason")}))


if __name__ == "__main__":
    main()
