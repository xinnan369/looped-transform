"""Fit count cutoffs on original training images and evaluate fixed checkpoints once."""

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from loopcrowd.calibration import (
    COUNT_CUTOFF, CUTOFF_GRID, DENSITY_GROUPS, FLOOR_DIAGNOSTICS, GRID_POINTS, SCORE_FLOOR, candidate_digest,
    choose_fit_records, count_match_metrics, density_group, fit_cutoff, fixed_pool_ap, json_digest,
    rank_metrics, strict_counts, summarize_counts,
)
from loopcrowd.data import load_rgb
from loopcrowd.experiment import file_sha, load_checkpoint, make_model, provenance, write_json
from loopcrowd.inference import predict_image
from loopcrowd.manifest import read_manifest
from loopcrowd.metrics import pooled_point_nap
from scripts.run_lc003 import MANIFEST_SHA, WEIGHT_SHA, load_json
from scripts.run_lc004a import BASELINES, SEEDS, TRAINING_CODE

FIT_MANIFEST = Path("reports/LC-004B/fit_manifest.json")
PROTOCOL = Path("reports/LC-004B/calibration_protocol.md")
ARCHIVE = Path("reports/LC-003/training_history.json")


def split_records(manifest):
    if file_sha(manifest) != MANIFEST_SHA:
        raise ValueError("the manifest must match the frozen ShanghaiTech Part A SHA-256")
    records = read_manifest(manifest)
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    if len(train) != 270 or len(val) != 30 or set(row.key for row in train) & set(row.key for row in val):
        raise ValueError("original train270/internal-val30 split was not preserved")
    boundaries = load_json(Path("reports/LC-003/diagnosis.json"))["heads_per_megapixel_train_tertiles"]
    recalculated = np.quantile([len(r.points) * 1e6 / (r.image_width * r.image_height) for r in train],
                               (1 / 3, 2 / 3))
    np.testing.assert_allclose(boundaries, recalculated, rtol=1e-13)
    fit, selected = choose_fit_records(records, boundaries)
    if any(not (manifest.parent / row.image_path).is_file() for row in selected + val):
        raise FileNotFoundError("the local calibration fit or validation images are incomplete")
    return fit, selected, val, boundaries


def checked_sources(root):
    archive = load_json(ARCHIVE)
    sources = {}
    for seed in SEEDS:
        for variant in BASELINES:
            key = f"warmup_cosine/{seed}/{variant}"
            source = root / "train" / key
            run = load_json(source / "run_manifest.json")
            expected = archive["runs"][key]["provenance"]
            config = run["config"]
            if (
                run != expected or json_digest(config) != run["config_sha256"]
                or run["git_commit"] != TRAINING_CODE or run["data"]["sha256"] != MANIFEST_SHA
                or run["weight_initialization"]["sha256"] != WEIGHT_SHA
                or config["count_threshold"] != COUNT_CUTOFF or config["localization_min_score"] != SCORE_FLOOR
                or run["completed_epochs"] != 40 or run["global_step"] != 5400 or run["execution"] != "completed"
            ):
                raise ValueError(f"LC-003 configuration or run provenance differs: {source}")
            checkpoint = source / "last.pt"
            if file_sha(checkpoint) != run["checkpoints"]["last.pt"]["sha256"]:
                raise ValueError(f"E40 checkpoint digest differs: {checkpoint}")
            sources[seed, variant] = (checkpoint, run)
    return sources


def fit_on_training(model, records, directory, config, device, calls):
    totals = np.zeros(len(CUTOFF_GRID), dtype=np.int64)
    snapshots = []
    gt = 0
    for record in records:
        if record.split != "train":
            raise ValueError("cutoff fit requires original training records")
        with load_rgb(record, directory) as image:
            predictions = predict_image(model, image, config, device, loop_count=calls)
        observed = strict_counts(predictions.count_candidates()[:, 2], CUTOFF_GRID)
        totals += np.abs(observed - len(record.points))
        snapshots.append([record.image_id, candidate_digest(predictions)])
        gt += len(record.points)
    cutoff, index = fit_cutoff(totals)
    return {
        "cutoff": cutoff, "fit_images": len(records), "fit_gt_heads": gt,
        "best_fit_absolute_error_sum": int(totals[index]),
        "raw_fit_absolute_error_sum": int(totals[GRID_POINTS.index(30)]),
        "fit_tied_minimizers": CUTOFF_GRID[totals == totals[index]].tolist(),
        "fit_grid_absolute_error_sums": totals.tolist(), "fit_candidate_sha256": json_digest(snapshots),
    }


def evaluate_validation(model, records, directory, config, device, boundaries, calls, cutoff):
    rows, ranked = [], []
    for record in records:
        if record.split != "val":
            raise ValueError("frozen evaluation requires internal-validation records")
        with load_rgb(record, directory) as image:
            predictions = predict_image(model, image, config, device, loop_count=calls)
        before = candidate_digest(predictions)
        raw, calibrated = count_match_metrics(record.points, predictions.count_candidates(), (COUNT_CUTOFF, cutoff))
        ranked.append(rank_metrics(record.points, predictions))
        after = candidate_digest(predictions)
        if before != after:
            raise AssertionError(f"count threshold evaluation changed the original pool for {record.image_id}")
        rows.append({
            "image_id": record.image_id, "density_group": density_group(record, boundaries),
            "gt": len(record.points), "raw": raw, "calibrated": calibrated,
            "candidate_rows": len(predictions.candidates), "owned_candidates": int(predictions.owned.sum()),
            "candidate_sha256": before,
        })

    def summary(samples, ranking):
        by_mode = {key: summarize_counts(samples, key) for key in ("raw", "calibrated")}
        ap = fixed_pool_ap(ranking, by_mode["raw"]["gt"], cutoff)
        return by_mode | {"fixed_pool_ap": ap}

    full = summary(rows, ranked)
    floors = {}
    for floor in FLOOR_DIAGNOSTICS:
        filtered_scores, filtered_matches = [], []
        for scores, matches in ranked:
            valid = scores > floor
            filtered_scores.append(scores[valid])
            filtered_matches.append(matches[:, valid])
        floors[f"{floor:.2f}"] = {
            "ap_candidates": sum(map(len, filtered_scores)),
            "local_nap": pooled_point_nap(filtered_scores, filtered_matches, full["raw"]["gt"]),
        }
    groups = {}
    for group in DENSITY_GROUPS:
        chosen = [(row, scores) for row, scores in zip(rows, ranked) if row["density_group"] == group]
        groups[group] = summary(*zip(*chosen)) if chosen else {"images": 0}
    sha = json_digest([[r["image_id"], r["candidate_sha256"]] for r in rows])
    return full | {
        "density_groups": groups, "per_image": rows, "raw_score_floor_diagnostics": floors,
        "validation_candidate_sha256": sha, "calibrated_candidate_sha256": sha,
    }


def verify_replay(result, seed, variant, calls):
    path = Path(f"reports/LC-004A/diagnostics/lc003/{seed}/{variant}/last.json")
    expected = load_json(path)["prefixes"][str(calls)]["overall"]
    for field in ("mae", "rmse", "predicted", "gt"):
        if result["raw"][field] != expected[field]:
            raise AssertionError(f"frozen raw {field} differs from LC-004A for seed {seed}/{variant}/{calls}")
    if result["fixed_pool_ap"]["original"] != expected["local_nap"]:
        raise AssertionError(f"frozen raw local nAP differs from LC-004A for seed {seed}/{variant}/{calls}")


def require_frozen_protocol():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracked = [str(FIT_MANIFEST), str(PROTOCOL), "src/loopcrowd/calibration.py", "scripts/run_lc004b.py"]
    subprocess.run(["git", "ls-files", "--error-unmatch", *tracked], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "HEAD", "--quiet", "--", *tracked], check=True)
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lc003", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-fit", action="store_true", help="Write the fit IDs before calibration runs")
    args = parser.parse_args()
    fit, fit_records, val_records, boundaries = split_records(args.manifest)
    if args.freeze_fit:
        if FIT_MANIFEST.exists():
            raise FileExistsError(f"fit manifest already frozen: {FIT_MANIFEST}")
        FIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        write_json(FIT_MANIFEST, fit)
        return
    if args.lc003 is None or args.output is None:
        parser.error("--lc003 and --output are required for the frozen experiment")
    revision = require_frozen_protocol()
    if fit != load_json(FIT_MANIFEST):
        raise ValueError("regenerated fit IDs differ from the frozen fit manifest")
    from loopcrowd.model import pretrained_path

    if file_sha(pretrained_path()) != WEIGHT_SHA:
        raise ValueError("the existing pretrained weight digest differs from the frozen archive")
    sources = checked_sources(args.lc003)
    cases = [(seed, variant, calls) for seed, variant in sources
             for calls in {"cnn_only": (0,), "shared_1": (1,), "shared_2": (0, 1, 2), "unshared_2": (2,)}[variant]]
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir()
    device = torch.device("cuda")
    write_json(args.output / "inputs.json", {
        "protocol_commit": revision, "protocol_sha256": file_sha(PROTOCOL),
        "manifest_path": str(args.manifest), "manifest_sha256": MANIFEST_SHA,
        "fit_file_sha256": file_sha(FIT_MANIFEST), "fit_ids_sha256": json_digest(fit),
        "val_ids_sha256": json_digest([[r.dataset, r.image_id] for r in val_records]),
        "train_images": 270, "fit_images": len(fit_records), "val_images": len(val_records),
        "val_ids": [r.image_id for r in val_records],
        "pretrained_path": str(pretrained_path()), "pretrained_sha256": WEIGHT_SHA,
        "train_density_boundaries": boundaries, "cutoff_grid": CUTOFF_GRID.tolist(),
        "count_cutoff": COUNT_CUTOFF, "fixed_score_floor": SCORE_FLOOR, "floor_diagnostics": FLOOR_DIAGNOSTICS,
    })

    def load_case(seed, variant):
        checkpoint, run = sources[seed, variant]
        state = load_checkpoint(checkpoint, torch.device("cpu"))
        if (state["config"] != run["config"] or state["data_sha256"] != MANIFEST_SHA
                or state["git_commit"] != TRAINING_CODE or state["history"][-1]["epoch"] != 40):
            raise ValueError(f"E40 checkpoint state differs from its frozen provenance: {checkpoint}")
        model = make_model(run["config"], initialize=False)
        model.load_state_dict(state["model"])
        return model.to(device).eval(), run

    fits = {}
    for seed, variant, calls in cases:
        start = perf_counter()
        name = f"{seed}-{variant}-{calls}"
        print(f"fit {name}", flush=True)
        model, run = load_case(seed, variant)
        result = fit_on_training(model, fit_records, args.manifest.parent, run["config"], device, calls)
        fits[name] = result | {"seconds": perf_counter() - start}
    fit_file = args.output / "frozen_thresholds.json"
    write_json(fit_file, fits)
    threshold_sha = file_sha(fit_file)
    fits = load_json(fit_file)

    for seed, variant, calls in cases:
        start = perf_counter()
        name = f"{seed}-{variant}-{calls}"
        print(f"validate {name}", flush=True)
        model, run = load_case(seed, variant)
        fit_result = fits[name]
        config = run["config"]
        result = evaluate_validation(model, val_records, args.manifest.parent, config, device,
                                     boundaries, calls, fit_result["cutoff"])
        verify_replay(result, seed, variant, calls)
        write_json(args.output / "cases" / f"{name}.json", {
            "execution": "completed", "seed": seed, "variant": variant, "calls": calls, "checkpoint_epoch": 40,
            "checkpoint": run["checkpoints"]["last.pt"], "config_sha256": run["config_sha256"],
            "training_code_sha": run["git_commit"], "frozen_thresholds_sha256": threshold_sha,
            "fit": fit_result, "validation": result, "validation_seconds": perf_counter() - start,
        })
    if file_sha(fit_file) != threshold_sha:
        raise AssertionError("frozen thresholds changed during validation")
    write_json(args.output / "run_manifest.json", provenance(run["config"], device, args.manifest) | {
        "execution": "completed", "cases": len(cases), "frozen_thresholds_sha256": threshold_sha,
    })
    print("LC-004B queue completed", flush=True)


if __name__ == "__main__":
    main()
