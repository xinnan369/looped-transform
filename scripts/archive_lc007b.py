"""Verify and archive the frozen LC-007B Stage 1 outputs, including partial evidence."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch

from loopcrowd.calibration import DENSITY_GROUPS, json_digest
from loopcrowd.experiment import file_sha, load_checkpoint, make_model, write_json
from scripts.run_lc007a import read_json, write_csv
from scripts.run_lc007b import FROZEN

ARTIFACTS = Path("reports/LC-007B")
POLICIES = ("live", "ema")
GROUPS = ("all", *DENSITY_GROUPS)
WINDOWS = {"late": tuple(range(36, 41)), "e40": (40,)}
PRIMARY_ARM = "shared_2_aux_025"
REFERENCES = ("cnn_only", "shared_1", "unshared_2_aux_025")
METRICS = ("mae", "rmse", "local_nap", "predicted", "signed_count_error", "tp", "fp", "fn")


def metrics(row):
    return {
        "images": row["images"], "gt": row["expected_heads"], "predicted": row["predicted_heads_on_labeled"],
        "signed_count_error": row["signed_count_error"], **row["count"], "local_nap": row["nap_local_v1"]["nap"],
    } | {short: row["local_" + short] for short in ("tp", "fp", "fn")}


def all_readouts(history):
    for row in history:
        live = {"final": row["validation"]}
        if "early_validation" in row:
            live["early"] = row["early_validation"]
        for policy, readouts in (("live", live), ("ema", row["ema_validation"])):
            for stage, raw in readouts.items():
                for group, result in {"all": raw, **raw["density_groups"]}.items():
                    yield {"epoch": row["epoch"], "policy": policy, "stage": stage, "density_group": group,
                           **metrics(result)}


def local_ref(path):
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def tracked_csv(directory, relative, rows):
    path = directory / relative
    write_csv(path, rows)
    return {"path": relative.as_posix(), "sha256": file_sha(path), "bytes": path.stat().st_size, "rows": len(rows)}


def flatten_epoch(row, digest):
    fields = ("epoch", "global_step", "optimizer_updates", "amp_skipped_updates", "ema_step", "crop_order_sha256",
              "crops", "sampled_gt", "overflow_gt", "positive_slots", "negative_slots", "train_seconds",
              "epoch_seconds", "mean_train_batch_ms", "peak_train_allocated_bytes", "peak_train_reserved_bytes",
              "lr_first", "lr_last", "lr_min", "lr_max", "loss", "classification_loss", "offset_loss",
              "final_loss", "early_loss", "early_classification_loss", "early_offset_loss", "gradient_l2_last_step",
              "image_metrics_sha256")
    return {key: row[key] for key in fields if key in row} | {
        f"gradient_l2_{key}": value for key, value in row["gradient_l2"].items()
    } | {"learning_rates_sha256": json_digest(row["learning_rates"]), "checkpoint_sha256": digest}


def load_image_metrics(run_dir, history, readouts, val_ids):
    output = []
    expected = {(r["epoch"], r["policy"], r["stage"], r["density_group"]): r for r in readouts}
    for row in history:
        epoch = row["epoch"]
        path = run_dir / "images" / f"{epoch:02d}.json"
        if file_sha(path) != row["image_metrics_sha256"]:
            raise ValueError(f"image evidence checksum mismatch: {path}")
        stage_rows = defaultdict(list)
        for item in read_json(path):
            if (item["gt"] != item["local_tp"] + item["local_fn"]
                    or item["predicted"] != item["local_tp"] + item["local_fp"]
                    or item["count_error"] != item["predicted"] - item["gt"]
                    or abs(item["count_error"]) != item["abs_count_error"]):
                raise ValueError(f"count/match identities differ: {path}")
            stage_rows[item["policy"], item["stage"]].append(item)
            output.append({"epoch": epoch, **item})
        if {(row["policy"], row["stage"]) for row in readouts if row["epoch"] == epoch} != stage_rows.keys():
            raise ValueError("per-image policy/stage has incomplete coverage")
        for (policy, stage), items in stage_rows.items():
            if [item["image_id"] for item in items] != val_ids:
                raise ValueError(f"validation image IDs differ: {path}")
            for group in GROUPS:
                subset = [item for item in items if group == "all" or item["density_group"] == group]
                aggregate = expected[epoch, policy, stage, group]
                if (len(subset) != aggregate["images"]
                        or not math.isclose(mean(item["abs_count_error"] for item in subset),
                                            aggregate["mae"], rel_tol=1e-12)
                        or not math.isclose(math.sqrt(mean(item["count_error"] ** 2 for item in subset)),
                                            aggregate["rmse"], rel_tol=1e-12)
                        or any(sum(item[key] for item in subset) != aggregate[metric]
                               for key, metric in (("gt", "gt"), ("predicted", "predicted"), ("count_error", "signed_count_error"),
                                                   ("local_tp", "tp"), ("local_fp", "fp"), ("local_fn", "fn")))):
                    raise ValueError(f"per-image and pooled count metrics differ: {path}")
    return output


def validate_saved_ema(path, run, latest):
    state = load_checkpoint(path, torch.device("cpu"))
    ema = state["ema"]
    if (state["config"] != run["config"] or state["git_commit"] != run["git_commit"]
            or state["data_sha256"] != run["data"]["sha256"] or state["history"][-1] != latest
            or state["scheduler"]["global_step"] != latest["global_step"]
            or ema["decay"] != .99 or ema["step"] != latest["ema_step"]):
        raise ValueError(f"saved parameter EMA provenance differs: {path}")
    expected = dict(make_model(run["config"], initialize=False).named_parameters())
    if ema["shadow"].keys() != expected.keys():
        raise ValueError(f"saved EMA parameter names differ: {path}")
    for name, value in ema["shadow"].items():
        if (value.dtype != torch.float32 or value.shape != expected[name].shape
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid saved FP32 EMA shadow tensor for {name}: {path}")


def case_artifacts(local, directory, seed, arm, state, preflight, phases, index):
    run_dir = local / "train" / str(seed) / arm
    path = run_dir / "run_manifest.json"
    details = {"seed": seed, "arm": arm, "status": state, "completed_epochs": 0, "updates": 0}
    if not path.is_file():
        return details, [], []
    run = read_json(path)
    details["manifest"] = local_ref(path)
    readouts, images = [], []
    if run.get("interrupted_checkpoint"):
        interrupted = run["interrupted_checkpoint"]
        path = Path(interrupted["path"])
        if file_sha(path) != interrupted["sha256"]:
            raise ValueError(f"interrupted checkpoint checksum differs: {path}")
        details["interrupted_checkpoint"] = local_ref(path)
    if run["execution"] in ("completed", "stopped"):
        details.update(completed_epochs=run["completed_epochs"], updates=run["global_step"],
                       config_sha256=run["config_sha256"], initial_state_sha256=run["initial_state_sha256"],
                       training_code_sha=run["git_commit"], manifest_sha256=run["data"]["sha256"],
                       pretrained_sha256=run["weight_initialization"]["sha256"])
    if any(p["case"] == f"{seed}/{arm}" for p in phases):
        matching = [p for p in phases if p["case"] == f"{seed}/{arm}"]
        details["gpu_seconds"] = sum(p["seconds"] for p in matching)
        details["peak_process_reserved_bytes"] = max(p["peak_reserved_bytes"] for p in matching)
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return details, [], []
    history = read_json(history_path)
    checkpoints = read_json(run_dir / "checkpoint_index.json")
    latest = run_dir / "last.pt"
    if (len(checkpoints) != len(history) or checkpoints[-1]["sha256"] != file_sha(latest)
            or run.get("checkpoints", {}).get("last.pt", {}).get("sha256", checkpoints[-1]["sha256"])
            != checkpoints[-1]["sha256"]):
        raise ValueError(f"last checkpoint index differs: {run_dir}")
    if (state == "completed" and len(history) != 40) or any(
        checkpoint["epoch"] != row["epoch"] or checkpoint["step"] != row["ema_step"]
        for checkpoint, row in zip(checkpoints, history)
    ):
        raise ValueError(f"completed history or checkpoint sequence differs: {run_dir}")
    validate_saved_ema(latest, run, history[-1])
    prefix = Path("epochs") / str(seed) / arm
    details.update(last_checkpoint=local_ref(latest), raw_history=local_ref(history_path),
                   crop_order_sha256=json_digest([row["crop_order_sha256"] for row in history]),
                   lr_sha256=json_digest([rate for row in history for rate in row["learning_rates"]]))
    readouts = list(all_readouts(history))
    images = load_image_metrics(run_dir, history, readouts, preflight["ordered_val_ids"])
    epochs = [flatten_epoch(row, ref["sha256"]) for row, ref in zip(history, checkpoints)]
    details["peak_train_allocated_bytes"] = max(row["peak_train_allocated_bytes"] for row in history)
    for kind, rows in (("training", epochs), ("metrics", readouts), ("images", images)):
        artifact = tracked_csv(directory, prefix / f"{kind}.csv", rows)
        index.append({"seed": seed, "arm": arm, "status": state, "kind": kind, **artifact})
    if state == "completed":
        left, right = (mean(row["final_loss"] for row in window) for window in (history[30:35], history[35:40]))
        details.update(final_loss_e31_35=left, final_loss_e36_40=right, loss_change=(right - left) / left,
                       finite=all(row["gradient_l2"]["nonfinite"] == 0 for row in history))
    return details, readouts, images


def endpoints(rows, *, window, policy, stage, group):
    selected = [row for row in rows if row["epoch"] in WINDOWS[window] and row["policy"] == policy
                and row["stage"] == stage and row["density_group"] == group]
    if len(selected) != len(WINDOWS[window]):
        raise ValueError("completed endpoint missing a frozen epoch")
    averages = {key: mean(row[key] for row in selected) for key in METRICS}
    averages["mae_cv"] = pstdev(row["mae"] for row in selected) / averages["mae"] if averages["mae"] else 0.0
    return averages


def paired_fields(a, b, *, left, right):
    return {f"{left}_{key}": a[key] for key in a} | {f"{right}_{key}": b[key] for key in b} | {
        f"delta_{key}": b[key] - a[key] for key in a
    }


def macro_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    for identity, group in groups.items():
        yield dict(zip(keys, identity)) | {"seed": "macro", "seed_count": len(group)} | {
            key: mean(row[key] for row in group) for key in group[0]
            if key not in ("seed", *keys) and type(group[0][key]) in (float, int)
        }


def policy_pairs(readouts, details):
    rows = []
    for (seed, arm), metrics_by_epoch in readouts.items():
        stages = tuple(dict.fromkeys(row["stage"] for row in metrics_by_epoch))
        for window in WINDOWS:
            for stage in stages:
                for group in GROUPS:
                    a, b = (endpoints(metrics_by_epoch, window=window, policy=policy, stage=stage, group=group)
                            for policy in POLICIES)
                    rows.append({"seed": seed, "arm": arm, "window": window, "stage": stage, "density_group": group,
                                 **paired_fields(a, b, left="live", right="ema")})
                    if (window, stage, group) == ("late", "final", "all"):
                        for policy, endpoint in zip(POLICIES, (a, b)):
                            details[seed, arm].update({f"{policy}_late_{key}": val for key, val in endpoint.items()})
    return rows


def late_images(rows, policy):
    images = defaultdict(list)
    for row in rows:
        if row["epoch"] in WINDOWS["late"] and row["policy"] == policy and row["stage"] == "final":
            images[row["image_id"]].append(row)
    return {key: (items[0]["density_group"], items[0]["gt"],
                  mean(row["abs_count_error"] for row in items),
                  mean(row["count_error"] for row in items)) for key, items in images.items()}


def architecture_pairs(readouts, images):
    rows, paired_images = [], []
    for seed in (42, 123):
        if (seed, PRIMARY_ARM) not in readouts:
            continue
        for arm in REFERENCES:
            if (seed, arm) not in readouts:
                continue
            per_image = {}
            for policy in POLICIES:
                a, b = (late_images(images[seed, candidate], policy) for candidate in (arm, PRIMARY_ARM))
                if a.keys() != b.keys() or any(a[key][:2] != b[key][:2] for key in a):
                    raise ValueError("independent-arm paired images differ")
                for key in a:
                    common = {"seed": seed, "reference": arm, "image_id": key, "density_group": a[key][0], "gt": a[key][1]}
                    per_image.setdefault(key, common).update({
                        f"{policy}_reference_mae": a[key][2], f"{policy}_shared_mae": b[key][2],
                        f"{policy}_mae_gap": b[key][2] - a[key][2], f"{policy}_signed_gap": b[key][3] - a[key][3],
                    })
            for row in per_image.values():
                row["mae_gap_change"] = row["ema_mae_gap"] - row["live_mae_gap"]
                paired_images.append(row)
            for window in WINDOWS:
                for group in GROUPS:
                    row = {"seed": seed, "reference": arm, "window": window, "density_group": group}
                    for policy in POLICIES:
                        ref, shared = (endpoints(readouts[seed, candidate], window=window, policy=policy, stage="final", group=group)
                                       for candidate in (arm, PRIMARY_ARM))
                        row.update({f"{policy}_{name}": value for name, value in
                                    paired_fields(ref, shared, left="reference", right="shared").items()})
                    row.update({f"gap_change_{key}": row[f"ema_delta_{key}"] - row[f"live_delta_{key}"] for key in METRICS})
                    if window == "late":
                        subset = [entry for entry in per_image.values() if group == "all" or entry["density_group"] == group]
                        for policy in POLICIES:
                            row.update({f"{policy}_{label}_image_fraction": mean(compare(entry[f"{policy}_mae_gap"]) for entry in subset)
                                        for label, compare in (("better", lambda x: x < 0), ("same", lambda x: x == 0),
                                                               ("worse", lambda x: x > 0))})
                    rows.append(row)
    return rows, paired_images


def decision(details, policy_rows, progress, frozen):
    limits = frozen["limits"]
    complete = [row for row in details.values() if row["status"] == "completed"]
    primary = [row for row in policy_rows if (row["window"], row["stage"], row["density_group"]) == ("late", "final", "all")]
    bounds = {}
    for arm in frozen["configs"]:
        paired = [row for row in primary if row["arm"] == arm]
        if len(paired) == 2:
            a, b = (mean(row[f"{policy}_mae"] for row in paired) for policy in POLICIES)
            delta_ap = mean(row["delta_local_nap"] for row in paired)
            bounds[arm] = {"live_mae": a, "ema_mae": b, "local_nap_change": delta_ap,
                           "mae_pass": b <= a * 1.03, "local_nap_pass": delta_ap >= -.005}
    checks = {
        "complete": progress["execution"] == "completed" and len(complete) == limits["runs"]
        and progress.get("successful_project_updates") == limits["total_updates"],
        "gpu_budget_pass": progress.get("gpu_seconds", 0) < limits["gpu_seconds"]
        and progress.get("peak_reserved_bytes", 0) <= limits["peak_reserved_bytes"],
        "all_ema_cv_pass": len(primary) == limits["runs"] and all(row["ema_mae_cv"] <= .10 for row in primary),
        "all_loss_finite_pass": len(complete) == limits["runs"] and all(
            row["finite"] and abs(row["loss_change"]) <= .03 for row in complete),
        "at_least_six_cv_improve": sum(row["delta_mae_cv"] < 0 for row in primary) >= 6,
        "all_arm_accuracy_guards_pass": len(bounds) == len(frozen["configs"]) and all(
            row["mae_pass"] and row["local_nap_pass"] for row in bounds.values()),
    }
    return {"screen_pass": all(checks.values()), "checks": checks, "per_arm": bounds,
            "cv_improved_runs": sum(row["delta_mae_cv"] < 0 for row in primary),
            "ema_cv_passing_runs": sum(row["ema_mae_cv"] <= .10 for row in primary),
            "recorded_decision_scope": "Stage 1 only", "queue_progress": progress}


def write_density_report(directory, architecture, policy):
    lines = [
        "# LC-007B density and paired error analysis", "",
        "The B train-only density tertiles are 82.6518 and 180.9862 heads/Mpx. The fixed validation groups contain",
        "13 low-density, 16 mid-density and 11 high-density images. The comparison uses the independently trained",
        "models within a seed and the same live or EMA policy. Local nAP is pooled inside each stratum for each",
        "epoch before averaging E36–40. Counts and matched TP/FP/FN use the strict owned count mask.", "",
    ]
    rows = [row for row in architecture if row["window"] == "late"]
    shared = [row for row in policy if (row["arm"], row["window"], row["stage"]) == (PRIMARY_ARM, "late", "final")]
    if shared:
        lines += [
            "Shared-2 aux's own EMA-minus-live E36–40 changes show whether a narrowed comparator gap",
            "coincided with an absolute regression in another stratum.", "",
            "| Stratum | Seeds | ΔMAE | ΔRMSE | Δ local nAP | ΔFP | ΔFN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in shared:
            lines.append(f"| {row['density_group']} | {row['seed_count']} | {row['delta_mae']:+.2f} | "
                         f"{row['delta_rmse']:+.2f} | {row['delta_local_nap']:+.5f} | "
                         f"{row['delta_fp']:+.1f} | {row['delta_fn']:+.1f} |")
        lines += [""]
    if not rows:
        lines += ["The queue did not supply any complete same-seed architecture contrast. Density effects are unavailable."]
    else:
        lines += [
            "Each entry is the Shared-2 aux value minus its matched comparator value. Negative MAE gaps favor",
            "Shared-2. Gap change is EMA gap minus live gap. The macro rows in the CSV average the available",
            "completed seeds with equal weight. Rows with one seed describe that seed only.", "",
            "| Comparator | Stratum | Seeds | MAE gap live → EMA (change) | RMSE gap live → EMA | nAP gap live → EMA |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['reference']} | {row['density_group']} | {row['seed_count']} | "
                f"{row['live_delta_mae']:+.2f} → {row['ema_delta_mae']:+.2f} ({row['gap_change_mae']:+.2f}) | "
                f"{row['live_delta_rmse']:+.2f} → {row['ema_delta_rmse']:+.2f} | "
                f"{row['live_delta_local_nap']:+.5f} → {row['ema_delta_local_nap']:+.5f} |")
        lines += ["", "| Comparator | Stratum | Signed count gap live → EMA | ΔTP live → EMA | ΔFP live → EMA | ΔFN live → EMA |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for row in rows:
            gaps = " | ".join(f"{row[f'live_delta_{key}']:+.1f} → {row[f'ema_delta_{key}']:+.1f}"
                               for key in ("signed_count_error", "tp", "fp", "fn"))
            lines.append(f"| {row['reference']} | {row['density_group']} | {gaps} |")
        by_reference = defaultdict(dict)
        for row in rows:
            by_reference[row["reference"]][row["density_group"]] = row
        lines += [""]
        for reference, groups in by_reference.items():
            high = groups["high"]
            direction = ("moved toward Shared-2" if high["gap_change_mae"] < 0 else
                         "moved toward the comparator" if high["gap_change_mae"] > 0 else "stayed the same")
            lines.append(
                f"- **{reference}.** The high-density MAE gap {direction} "
                f"({high['gap_change_mae']:+.2f} heads). The low- and mid-density gap changes were "
                f"{groups['low']['gap_change_mae']:+.2f} and {groups['mid']['gap_change_mae']:+.2f}.")
        lines += [
            "", "The [architecture table](architecture_comparison.csv) retains both seeds, secondary E40",
            "rows and paired fractions of images with better, unchanged and worse late absolute error. The",
            "[image pairs](architecture_image_pairs.csv) retain the joined IDs, both policies' per-image late",
            "MAE and signed gaps, and the EMA-minus-live change. Repeated readings of the same validation",
            "images and the two training seeds are descriptive development evidence.",
        ]
    (directory / "density_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    source, directory = args.local, args.report
    directory.mkdir(parents=True, exist_ok=True)
    frozen = read_json(FROZEN)
    progress = read_json(source / "queue_progress.json")
    if progress["execution"] == "in_progress":
        parser.error("archive only a completed or stopped queue")
    preflight_file = source / "preflight.json"
    preflight = read_json(preflight_file) if preflight_file.is_file() else {
        "execution": "not_run", "reason": progress.get("reason", "preflight was unavailable")}
    if preflight["execution"] == "passed" and file_sha(preflight_file) != progress.get("preflight_sha256"):
        raise ValueError("frozen preflight checksum differs from the execution ledger")
    fixture = source / "synthetic_validation.png"
    if fixture.is_file():
        preflight["generated_validation_fixture"] = local_ref(fixture)
    write_json(directory / "preflight.json", preflight)
    phase_file = source / "gpu_phases.jsonl"
    phases = [json.loads(line) for line in phase_file.read_text(encoding="utf-8").splitlines()] if phase_file.is_file() else []
    if phases and (not math.isclose(sum(row["seconds"] for row in phases), progress["gpu_seconds"], rel_tol=1e-12)
                   or phases[-1]["successful_project_updates"] != progress["successful_project_updates"]):
        raise ValueError("GPU phase ledger total differs from queue")
    if phases:
        write_csv(directory / "gpu_phases.csv", phases)
    completed = {(row["seed"], row["arm"]) for row in progress["ordered_runs"]}
    index, readouts, details, images = [], {}, {}, {}
    for seed, arms in frozen["order"].items():
        for arm in arms:
            key = int(seed), arm
            status = "completed" if key in completed else "not_run"
            if status != "completed" and progress.get("current_run") == f"{seed}/{arm}":
                status = "stopped"
            details[key], metrics_by_epoch, image_rows = case_artifacts(
                source, directory, *key, status, preflight, phases, index)
            if status == "completed":
                readouts[key], images[key] = metrics_by_epoch, image_rows
            if not metrics_by_epoch:
                index.append({"seed": seed, "arm": arm, "status": status, "kind": "unavailable"})
    policy = policy_pairs(readouts, details)
    architecture, image_pairs = architecture_pairs(readouts, images)
    write_csv(directory / "epoch_index.csv", index)
    write_csv(directory / "experiment_matrix.csv", list(details.values()))
    policy_macro = list(macro_rows(policy, ("arm", "window", "stage", "density_group")))
    write_csv(directory / "paired_live_ema.csv", policy + policy_macro)
    architecture_macro = list(macro_rows(architecture, ("reference", "window", "density_group")))
    write_csv(directory / "architecture_comparison.csv", architecture + architecture_macro)
    if image_pairs:
        write_csv(directory / "architecture_image_pairs.csv", image_pairs)
    write_json(directory / "research_decision.json", decision(details, policy, progress, frozen))
    write_density_report(directory, architecture_macro, policy_macro)


if __name__ == "__main__":
    main()
