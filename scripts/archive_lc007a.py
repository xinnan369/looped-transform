"""Archive LC-007A historical recomputations and the frozen paired replay."""

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from loopcrowd.accuracy_diagnosis import bootstrap_indices, paired_rows, sampling_sensitivity
from loopcrowd.calibration import DENSITY_GROUPS, json_digest
from loopcrowd.experiment import file_sha, write_json
from scripts.run_lc007a import ROOT, SEEDS, archive_runs, planned_cases, read_csv, read_json, write_csv

STAGES = ("early", "final")
GROUPS = ("all", *DENSITY_GROUPS)
AUX = "shared_2_aux_025"
SHARED_FINAL = "shared_2_final_only"
UNSHARED_AUX = "unshared_2_aux_025"
UNSHARED_FINAL = "unshared_2_final_only"
METRICS = ("mae", "rmse", "local_nap", "gt", "predicted", "tp", "fp", "fn")


def compare_arms(dataset):
    comparisons = [(AUX, arm) for arm in (SHARED_FINAL, "cnn", "shared_1")]
    comparisons.append((SHARED_FINAL, UNSHARED_FINAL))
    if dataset == "B":
        comparisons.extend(((AUX, UNSHARED_AUX), (UNSHARED_AUX, UNSHARED_FINAL), (UNSHARED_AUX, "cnn")))
    for target, reference in comparisons:
        yield (target, "final"), (reference, "final")
    staged = (SHARED_FINAL, AUX) if dataset == "A" else (SHARED_FINAL, AUX, UNSHARED_FINAL, UNSHARED_AUX)
    for arm in staged:
        yield (arm, "final"), (arm, "early")


def contrast_name(target, reference):
    return f"{target[0]}_{target[1]}--{reference[0]}_{reference[1]}"


def design(dataset, target, reference):
    if target[0] == reference[0]:
        return "within_checkpoint"
    if dataset == "A" and {target[0], reference[0]} != {AUX, SHARED_FINAL}:
        return "cross_archive_context"
    return "independently_trained_matched_recipe"


def artifact_ref(path, directory):
    return {"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def density_summary(pairs, target, reference):
    if not pairs:
        return {"images": 0}
    changes = np.array([row["delta_absolute_error"] for row in pairs])
    result = {
        "images": len(pairs), "improved_images": int((changes < 0).sum()),
        "worse_images": int((changes > 0).sum()), "tied_images": int((changes == 0).sum()),
        "improved_fraction": float((changes < 0).mean()), "worse_fraction": float((changes > 0).mean()),
        "tied_fraction": float((changes == 0).mean()),
    }
    for key in METRICS:
        a, b = target[key], reference[key]
        result.update({f"target_{key}": a, f"reference_{key}": b,
                       f"delta_{key}": a - b if a is not None and b is not None else None})
    return result


def metric_snapshot(epoch, stage):
    evaluation = epoch.get("validation_stages", {}).get(stage)
    if evaluation is None:
        evaluation = epoch.get("validation_" + stage)
    if not evaluation:
        evaluation = epoch["validation"] if stage == "final" else None
    if not evaluation:
        return {}
    return {
        "mae": evaluation["count"]["mae"], "rmse": evaluation["count"]["rmse"],
        "local_nap": evaluation["nap_local_v1"]["nap"], "predicted": evaluation["predicted_heads_on_labeled"],
        "gt": evaluation["expected_heads"],
    } | {key: evaluation["local_" + key] for key in ("tp", "fp", "fn") if "local_" + key in evaluation}


def recompute_history(run_id, run):
    # LC-003 logs predate staged epoch tables. The later tracked epoch CSVs
    # contain the complete primary fields regardless of local raw-log availability.
    series = []
    p = run["provenance"]
    if run_id.startswith("LC-003/"):
        for epoch in run["epochs"]:
            series.append({
                "epoch": epoch["epoch"], "optimizer_loss": epoch["loss"], "final_loss": epoch["loss"],
                "final_classification_loss": epoch["classification_loss"], "final_offset_loss": epoch["offset_loss"],
                "gradient_l2_mean": epoch["gradient_l2"]["mean"],
                "gradient_l2_nonfinite": epoch["gradient_l2"]["nonfinite"],
            } | {key: epoch[key] for key in ("crops", "sampled_gt", "overflow_gt", "positive_slots", "negative_slots",
                                             "optimizer_updates", "amp_skipped_updates", "lr_first", "lr_last",
                                             "lr_min", "lr_max", "epoch_seconds", "train_seconds")}
                | {"final_" + key: value for key, value in metric_snapshot(epoch, "final").items()})
    elif run_id.startswith("LC-005/"):
        _, seed, arm = run_id.split("/")
        series = [row for row in read_csv(ROOT / "LC-005/training_history.csv")
                  if (row["seed"], row["arm"]) == (seed, arm)]
    else:
        _, seed, arm = run_id.split("/")
        series = read_csv(ROOT / "LC-006/epochs" / seed / f"{arm}.csv")
    if [int(row["epoch"]) for row in series] != list(range(1, 41)):
        raise ValueError(f"historical epoch sequence differs for {run_id}")
    if sum(int(row["optimizer_updates"]) for row in series) != p["global_step"]:
        raise ValueError(f"historical update count differs for {run_id}")
    return series


def correlation(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def historical_summary(run_id, series, run):
    def values(key):
        return np.array([float(row[key]) for row in series], dtype=float)

    mae, loss = values("final_mae"), values("final_loss")
    late_mae = mae[-5:].mean()
    cv = mae[-5:].std() / late_mae if late_mae else 0
    loss_change = (loss[-5:].mean() - loss[-10:-5].mean()) / loss[-10:-5].mean()
    fields = ("optimizer_loss", "final_loss", "final_classification_loss", "final_offset_loss")
    if series[0].get("early_loss") not in (None, ""):
        fields += ("early_loss", "early_classification_loss", "early_offset_loss")
    finite = bool(all(np.isfinite(values(key)).all() for key in fields)
                  and not values("gradient_l2_nonfinite").any())
    stable = bool(cv <= .10 and abs(loss_change) <= .03 and finite)
    archived = run["summary"]
    if run_id.startswith("LC-003/"):
        expected = (archived["late_means"]["mae"], archived["late_means"]["rmse"],
                    archived["late_means"]["local_nap"], archived["late_mae_cv"], archived["late_loss_change"])
    else:
        loss_key = "final_loss_late_relative_change" if run_id.startswith("LC-005/") else "final_late_loss_change"
        expected = tuple(archived["final_late_" + key] for key in METRICS[:3]) + (
            archived["final_late_mae_cv"], archived[loss_key])
    np.testing.assert_allclose((late_mae, values("final_rmse")[-5:].mean(), values("final_local_nap")[-5:].mean(),
                                cv, loss_change), expected, rtol=1e-12, atol=1e-12, err_msg=run_id)
    if stable != archived["stability_pass"]:
        raise ValueError(f"recomputed historical gate differs for {run_id}")
    result = {"run": run_id, "mae_cv": float(cv), "final_loss_change": float(loss_change),
              "finite": finite, "stability_pass": stable, "epochs": len(series),
              "updates": sum(int(row["optimizer_updates"]) for row in series),
              "epochs_seconds": float(values("epoch_seconds").sum()),
              "train_seconds": float(values("train_seconds").sum()),
              "min_lr": float(values("lr_min").min()), "max_lr": float(values("lr_max").max()),
              "last_lr": float(values("lr_last")[-1]),
              "amp_skipped": int(values("amp_skipped_updates").sum())}
    for stage in STAGES:
        for key in METRICS:
            field = archived_metric_field(series, stage, key)
            if series[0].get(field) not in (None, ""):
                data = values(field)
                result[f"{stage}_late_{key}"] = float(data[-5:].mean())
                result[f"{stage}_e40_{key}"] = float(data[-1])
    for key in fields + ("sampled_gt", "overflow_gt", "positive_slots", "negative_slots", "gradient_l2_mean",
                         "final_predicted", "final_local_nap",
                         "early_predicted", "early_local_nap", "early_local_tp", "early_local_fp", "early_local_fn",
                         "final_local_tp", "final_local_fp", "final_local_fn"):
        if series[0].get(key) not in (None, ""):
            data = values(key)
            result.update({f"{key}_e1": float(data[0]), f"{key}_e40": float(data[-1]),
                           f"{key}_e36_40_mean": float(data[-5:].mean()),
                           f"{key}_e36_40_min": float(data[-5:].min()),
                           f"{key}_e36_40_max": float(data[-5:].max()),
                           f"{key}_late_r_with_mae": correlation(data[-5:], mae[-5:]),
                           f"{key}_all_epoch_r_with_mae": correlation(data, mae)})
    if series[0].get("peak_train_allocated_bytes") not in (None, ""):
        for key in ("peak_train_allocated_bytes", "peak_train_reserved_bytes"):
            result[key] = int(values(key).max())
    return result


def archived_metric_field(series, stage, key):
    column = f"{stage}_{key}"
    if column not in series[0] and key in ("tp", "fp", "fn"):
        column = f"{stage}_local_{key}"
    return column


def archive_historical(directory):
    summaries, series_by_run = {}, {}
    for run_id, (run, _, _) in sorted(archive_runs().items()):
        if run_id.startswith("LC-004A/"):
            continue
        series = recompute_history(run_id, run)
        summaries[run_id] = historical_summary(run_id, series, run)
        series_by_run[run_id] = series
    write_csv(directory / "historical_recomputation.csv", list(summaries.values()))
    comparisons = []

    def add(group, seed, target, reference, target_stage="final", reference_stage="final"):
        a, b = summaries[target], summaries[reference]
        row = {"comparison": group, "seed": seed, "target": target, "reference": reference,
               "target_stage": target_stage, "reference_stage": reference_stage}
        for key in METRICS[:3]:
            x, y = a[f"{target_stage}_late_{key}"], b[f"{reference_stage}_late_{key}"]
            row.update({f"target_{key}": x, f"reference_{key}": y, f"delta_{key}": x - y})
        comparisons.append(row)

    for arm in ("cnn_only", "shared_1", "shared_2"):
        add("LC003_A_schedule_pilot", 42, f"LC-003/warmup_cosine/42/{arm}", f"LC-003/constant/42/{arm}")
    for seed in SEEDS:
        for target, reference in (("shared_1", "cnn_only"), ("shared_2", "shared_1"), ("shared_2", "unshared_2")):
            add("LC003_A_matched_cosine", seed, f"LC-003/warmup_cosine/{seed}/{target}",
                f"LC-003/warmup_cosine/{seed}/{reference}")
        for stage in STAGES:
            add("LC005_A_supervision", seed, f"LC-005/{seed}/aux_025", f"LC-005/{seed}/final_only", stage, stage)
        for arm in ("aux_025", "final_only"):
            add("LC005_A_within_checkpoint", seed, f"LC-005/{seed}/{arm}", f"LC-005/{seed}/{arm}", "final", "early")
        for target, reference in compare_arms("B"):
            add("LC006_B_" + design("B", target, reference), seed, f"LC-006/{seed}/{target[0]}",
                f"LC-006/{seed}/{reference[0]}", target[1], reference[1])
        for target, reference in ((AUX, SHARED_FINAL), (UNSHARED_AUX, UNSHARED_FINAL)):
            add("LC006_B_aux_early", seed, f"LC-006/{seed}/{target}", f"LC-006/{seed}/{reference}", "early", "early")
    write_csv(directory / "historical_paired_late.csv", comparisons)
    return series_by_run


def read_image_rows(path):
    return [{key: int(value) if key in ("width", "height", "gt", "predicted", "signed_error", "absolute_error",
                                       "tp", "fp", "fn", "covered_raw_floor", "covered_owned_floor",
                                       "covered_count_mask", "fn_near_seam", "fp_near_seam") else value
             for key, value in row.items()} for row in read_csv(path)]


def archive_replay(local, directory, histories):
    evaluated = read_json(local / "evaluation.json")
    actual = [(case["dataset"], case["seed"], case["arm"], case["historical_run"], case["staged"])
              for case in evaluated["cases"]]
    if actual != list(planned_cases()):
        raise ValueError("replay manifest does not account for the full frozen plan")
    readouts, index, reconciliation, stage_groups = {}, [], [], []
    for case in evaluated["cases"]:
        data, seed, arm = case["dataset"], case["seed"], case["arm"]
        base = {"dataset": data, "seed": seed, "arm": arm, "e40_sha256": case["e40_sha256"],
                "status": case["status"], "reason": case.get("reason", "")}
        if case["status"] != "completed":
            index.append(base | {"kind": "stage"})
            continue
        if case["full_internal_val_passes"] != 1:
            raise ValueError("expected one full pass per completed checkpoint")
        for stage, metrics in case["stages"].items():
            source = local / metrics["csv"]
            if file_sha(source) != metrics["sha256"]:
                raise ValueError(f"stage evidence digest mismatch: {source}")
            rows = read_image_rows(source)
            ids = [row["image_id"] for row in rows]
            if ids != evaluated["datasets"][data]["ordered_val_ids"]:
                raise ValueError(f"incomplete validation image alignment in {source}")
            readouts[data, seed, arm, stage] = rows, metrics
            path = directory / "images" / metrics["csv"]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)
            index.append(base | {"kind": "stage", "stage": stage, "rows": len(rows)} | artifact_ref(path, directory))
            series = histories[case["historical_run"]]
            archived = {key: float(series[-1][column]) for key in METRICS
                        if (column := archived_metric_field(series, stage, key)) in series[-1]}
            delta = {key: metrics["all"][key] - value for key, value in archived.items()}
            reconciliation.append(base | {"stage": stage} | delta | {
                "matches_archived_metrics": all(abs(change) <= 1e-12 for change in delta.values()),
            })
            for group in GROUPS:
                group_row = base | {"stage": stage, "density_group": group} | metrics[group]
                if group == "all":
                    top = metrics["largest_absolute_errors"]
                    denominator = metrics["absolute_error_sum"]
                    group_row.update({"top5": top, "absolute_error_sum": denominator,
                                      "top5_fraction": sum(row["absolute_error"] for row in top) / denominator
                                      if denominator else 0})
                stage_groups.append(group_row)
    write_csv(directory / "e40_reconciliation.csv", reconciliation)
    write_csv(directory / "e40_groups.csv", stage_groups)
    return evaluated, readouts, index


def archive_pairs(directory, evaluated, readouts, index):
    summary, sensitivity = [], {}
    for dataset, scope in evaluated["datasets"].items():
        if scope["status"] != "verified":
            sensitivity[dataset] = scope
            continue
        ids = scope["ordered_val_ids"]
        indices = bootstrap_indices(len(ids))
        sensitivity[dataset] = {
            "ordered_image_ids": ids, "shared_indices_sha256_little_endian_int64":
                hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
            "seed": 7007, "resamples": len(indices), "rng": "numpy.PCG64", "contrasts": {},
        }
        for target, reference in compare_arms(dataset):
            name = contrast_name(target, reference)
            diffs, available = [], []
            by_group = defaultdict(list)
            for seed in SEEDS:
                category = design(dataset, target, reference)
                meta = {"dataset": dataset, "seed": seed, "contrast": name, "comparison_type": category}
                a = readouts.get((dataset, seed, *target))
                b = readouts.get((dataset, seed, *reference))
                if a is None or b is None:
                    index.append(meta | {"kind": "pair", "status": "not_run", "reason": "missing stage evidence"})
                    continue
                pairs = paired_rows(a[0], b[0])
                path = directory / "pairs" / dataset / str(seed) / f"{name}.csv"
                write_csv(path, pairs)
                index.append(meta | {"kind": "pair", "status": "completed", "rows": len(pairs)}
                             | artifact_ref(path, directory))
                for group in GROUPS:
                    selected = pairs if group == "all" else [row for row in pairs if row["density_group"] == group]
                    data = density_summary(selected, a[1][group], b[1][group])
                    summary.append(meta | {"density_group": group} | data)
                    by_group[group].append(data)
                available.append(seed)
                diffs.append([row["delta_absolute_error"] for row in pairs])
            for group, rows in by_group.items():
                macro = {}
                if rows[0]["images"]:
                    macro = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
                             if key != "images" and all(row[key] is not None for row in rows)}
                summary.append({"dataset": dataset, "seed": "macro", "contrast": name,
                                "comparison_type": design(dataset, target, reference), "density_group": group,
                                "images": rows[0]["images"], "seed_count": len(rows)} | macro)
            if available:
                sensitivity[dataset]["contrasts"][name] = {
                    "paired_training_seeds": available, **sampling_sensitivity(diffs, indices),
                }
    write_csv(directory / "paired_image_metrics.csv", index)
    write_csv(directory / "paired_summary.csv", summary)
    write_json(directory / "sampling_sensitivity.json", sensitivity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "LC-007A")
    args = parser.parse_args()
    histories = archive_historical(args.report)
    evaluated, readouts, index = archive_replay(args.local, args.report, histories)
    archive_pairs(args.report, evaluated, readouts, index)
    for filename in ("frozen_cases.csv", "run_ledger.csv"):
        shutil.copyfile(args.local / filename, args.report / filename)
    evidence = evaluated | {"local_evaluation_sha256": file_sha(args.local / "evaluation.json")}
    # The device-level BN statistics are compacted to an auditable buffer digest
    # and tracked batch counts. Full layer summaries stay with the replay output.
    for case in evidence["cases"]:
        if case["status"] != "completed":
            continue
        buffers = case.pop("bn_buffers")
        case["bn_statistics_sha256"] = json_digest(buffers)
        case["bn_batches_tracked"] = sorted({layer["num_batches_tracked"] for layer in buffers.values()})
        case.pop("stages")
    write_json(args.report / "replay_execution.json", evidence)


if __name__ == "__main__":
    main()
