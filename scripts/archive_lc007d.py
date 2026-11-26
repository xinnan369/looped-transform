"""Verify local LC-007D results and archive the frozen paired comparisons on CPU."""

import argparse
import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from shutil import copyfile
from statistics import mean, stdev

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, write_json
from loopcrowd.model import CHANNEL_GATES
from scripts.archive_lc007b import (
    GROUPS, METRICS, all_readouts, flatten_epoch, load_image_metrics, local_ref, tracked_csv, validate_saved_ema,
)
from scripts.archive_lc007c import POLICIES, SEEDS, SHARED, UNSHARED, correction, load_readouts, read_frozen_files
from scripts.prepare_lc007d import CONFIG
from scripts.run_lc007a import read_json, write_csv
from scripts.run_lc007d import require_authorization

CONTRASTS = (
    ("RS", SHARED, "primary"), ("R", SHARED, "residual_ablation"), ("RS", "R", "state_increment"),
    ("R", "cnn_only", "independent_cnn"), ("RS", "cnn_only", "independent_cnn"),
    ("R", "shared_1", "independent_one_call"), ("RS", "shared_1", "independent_one_call"),
    ("R", UNSHARED, "ungated_unshared_target"), ("RS", UNSHARED, "ungated_unshared_target"),
    ("RS", "U-RS", "matched_sharing"),
)
JOINT_GATES = {"primary", "residual_ablation", "matched_sharing"}


def close_out(root, config, *, config_path=CONFIG):
    run = root / "run"
    code = int((root / "exit_status.txt").read_text(encoding="utf-8"))
    provenance = {key: (root / f"{key}.txt").read_text(encoding="utf-8").strip()
                  for key in ("launch_time", "exit_time")}
    authorization = read_json(root / "authorization.json")
    revision = authorization["code_commit"]
    require_authorization(authorization, config, file_sha(config_path), revision)
    path = run / "queue_progress.json"
    if path.is_file():
        progress = read_json(path)
    elif code:
        progress = {"execution": "stopped", "completed": [], "code_commit": revision,
                    "reason": "runner exited before writing queue progress"}
    else:
        raise ValueError("finished runner did not record its resource ledger")
    if progress["code_commit"] != revision or (progress["execution"] == "completed") != (code == 0):
        raise ValueError("queue status, code revision or process exit status differs")
    path = run / "gpu_phases.jsonl"
    phases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []
    limits = config["limits_if_approved"]
    peak_limit = limits.get("peak_reserved_bytes")
    if progress["execution"] == "completed":
        order = [f"{seed}/{arm}" for seed, arms in config["order"].items() for arm in arms]
        if (progress["completed"] != order or progress["successful_updates"] != limits["total_updates"]
                or progress["charged_seconds"] > limits["gpu_seconds"]
                or (peak_limit is not None and progress["peak_reserved_bytes"] > peak_limit)
                or not phases or phases[-1]["phase"] != "cleanup" or any(row["failure"] for row in phases)):
            raise ValueError("completed queue does not reconcile to the scoped resource ledger")
    return authorization, progress, phases, provenance | {"exit_status": code, "runner_log": local_ref(root / "runner.log")}


def verify_case(path, seed, arm, config, preflight, reference, complete, revision, *, depth_calls=2):
    run_path = path / "run_manifest.json"
    history_path = path / "history.json"
    evidence = {"seed": seed, "arm": arm, "status": "not_run", "epochs": 0, "successful_updates": 0}
    if not run_path.is_file():
        return evidence, [], [], [], []
    evidence.update(status="stopped", manifest=local_ref(run_path))
    run = read_json(run_path)
    if run.get("interrupted_checkpoint"):
        interrupted = run["interrupted_checkpoint"]
        partial = local_ref(Path(interrupted["path"]))
        if interrupted["sha256"] != partial["sha256"]:
            raise ValueError(f"interrupted checkpoint digest differs for {seed}/{arm}")
        evidence["interrupted_checkpoint"] = partial
    if not history_path.is_file():
        return evidence, [], [], [], []
    history = read_json(history_path)
    checkpoints = read_json(path / "checkpoint_index.json")
    last = path / "last.pt"
    planned = preflight["planned_starts" if depth_calls > 2 else "planned_gated_initialization"][f"{seed}/{arm}"]
    if (run["git_commit"] != revision or run["tracked_changes"] != "" or run["config_sha256"] != planned["config_sha256"]
            or json_digest(run["config"]) != planned["config_sha256"]
            or run["initial_state_sha256"] != planned["initial_sha256"]
            or run["data"]["sha256"] != preflight["data"]["manifest_sha256"]
            or run["weight_initialization"]["sha256"] != preflight["data"]["pretrained_weight_sha256"]
            or run["ema"]["density_tertiles"] != preflight["data"]["train_only_density_tertiles"]
            or run["ema"]["buffers"] != "live"):
        raise ValueError(f"run provenance differs for {seed}/{arm}")
    if not history or len(checkpoints) != len(history) or checkpoints[-1]["sha256"] != file_sha(last):
        raise ValueError(f"checkpoint and epoch index differ for {seed}/{arm}")
    if complete and (len(history) != config["limits_if_approved"]["epochs_per_run"]
                     or run["execution"] != "completed" or run["global_step"] != history[-1]["global_step"]):
        raise ValueError(f"completed run horizon differs for {seed}/{arm}")
    gates = []
    arm_config = config["arms" if depth_calls > 2 else "variants"][arm]
    branches = CHANNEL_GATES.get(arm_config["variant"], ())
    expected_gate_keys = {(stage, branch) for stage in range(1, depth_calls + 1) for branch in branches}
    for epoch, (row, ckpt, ref) in enumerate(zip(history, checkpoints, reference[:len(history)], strict=True), 1):
        if (row["epoch"] != epoch or ckpt["epoch"] != epoch or row["optimizer_updates"] != 180
                or row["global_step"] != 180 * epoch or row["ema_step"] != 180 * epoch or ckpt["step"] != 180 * epoch
                or row["amp_skipped_updates"] or row["gradient_l2"]["nonfinite"]
                or row["crop_order_sha256"] != ref["crop_order_sha256"]
                or len(row["learning_rates"]) != 180 or json_digest(row["learning_rates"]) != ref["learning_rates_sha256"]):
            raise ValueError(f"optimizer, EMA, crop or LR trajectory differs for {seed}/{arm}/{epoch}")
        stats = row.get("channel_gate_statistics", {})
        if set(stats) != (set(POLICIES) if branches else set()):
            raise ValueError(f"gate policies missing for {seed}/{arm}/{epoch}")
        for policy, rows in stats.items():
            if len(rows) != len(expected_gate_keys) or {(r["stage"], r["branch"]) for r in rows} != expected_gate_keys:
                raise ValueError(f"gate branches missing for {seed}/{arm}/{epoch}/{policy}")
            for item in rows:
                if (not all(isfinite(item[k]) for k in ("min", "max", "mean", "l2", "delta_l2"))
                        or not item["min"] - 1e-6 <= item["mean"] <= item["max"] + 1e-6):
                    raise ValueError(f"invalid gate statistics for {seed}/{arm}/{epoch}/{policy}")
                gates.append({"seed": seed, "arm": arm, "epoch": epoch, "policy": policy, **item})
    validate_saved_ema(last, run, history[-1])
    readouts = list(all_readouts(history))
    images = load_image_metrics(path, history, readouts, preflight["data"]["ordered_val_ids"])
    names = {depth_calls // 2: "early", depth_calls: "final"}
    stages = tuple(names.get(call, f"stage{call}") for call in range(1, depth_calls + 1))
    first_full_epoch = run["config"].get("depth_readout_start_epoch", 1)
    expected = {(e, p, s, g) for e in range(1, len(history) + 1) for p in POLICIES
                for s in (stages if e >= first_full_epoch else ("early", "final")) for g in GROUPS}
    if len(readouts) != len(expected) or {(r["epoch"], r["policy"], r["stage"], r["density_group"])
                                           for r in readouts} != expected:
        raise ValueError(f"missing metric readouts for {seed}/{arm}")
    if any(not 0 <= row["local_nap"] <= 1 for row in readouts):
        raise ValueError(f"invalid pooled local nAP for {seed}/{arm}")
    training = [flatten_epoch(row, ckpt["sha256"]) for row, ckpt in zip(history, checkpoints, strict=True)]
    evidence.update(status="completed" if complete else "stopped", epochs=len(history),
                    successful_updates=run.get("global_step", history[-1]["global_step"]),
                    retained_checkpoint=local_ref(last), raw_history=local_ref(history_path))
    return evidence, training, readouts, images, gates


def endpoint(table, epochs, policy, stage, group):
    expected = {(e, policy, stage, group) for e in epochs}
    rows = [row for row in table if (int(row["epoch"]), row["policy"], row["stage"], row["density_group"]) in expected]
    if len(rows) != len(epochs):
        raise ValueError("incomplete endpoint")
    return {metric: mean(float(row[metric]) for row in rows) for metric in METRICS}


def stability(training, readouts, decision):
    final = [float(row["mae"]) for row in readouts if row["policy"] == "ema" and row["stage"] == "final"
             and row["density_group"] == "all" and int(row["epoch"]) in decision["analysis_epochs"]]
    losses = {int(row["epoch"]): float(row["final_loss"]) for row in training}
    before, after = (mean(losses[epoch] for epoch in decision[key])
                     for key in ("loss_reference_epochs", "analysis_epochs"))
    cv = stdev(final) / mean(final) if mean(final) > 0 else None
    loss_change = (after - before) / before if before > 0 else None
    passed = (cv is not None and cv <= decision["late_mae_sample_cv_max"] and loss_change is not None
              and abs(loss_change) <= decision["absolute_relative_final_loss_change_max"])
    return {"sample_mae_cv": cv, "relative_final_loss_change": loss_change,
            "final_loss_reference": before, "final_loss_late": after, "passed": passed}


def paired_rows(readouts, decision):
    rows = []
    for candidate, reference, label in CONTRASTS:
        for window, epochs in (("E36-E40", decision["analysis_epochs"]), ("E40", (40,))):
            for policy in POLICIES:
                stages = ("final",) if reference in ("cnn_only", "shared_1") else ("early", "final")
                for stage in stages:
                    for group in GROUPS:
                        scope = {"candidate": candidate, "reference": reference, "contrast": label, "policy": policy,
                                 "stage": stage, "density_group": group, "window": window}
                        paired = []
                        for seed in SEEDS:
                            if (seed, candidate) not in readouts or (seed, reference) not in readouts:
                                continue
                            cand, ref = [endpoint(readouts[seed, arm], epochs, policy, stage, group)
                                         for arm in (candidate, reference)]
                            values = {f"{prefix}_{key}": val for prefix, metrics in (("candidate", cand), ("reference", ref))
                                      for key, val in metrics.items()} | {f"delta_{key}": cand[key] - ref[key] for key in METRICS}
                            paired.append(values)
                            rows.append(scope | {"seed": seed, "seed_count": 1, **values, "status": "measured"})
                        if len(paired) == len(SEEDS):
                            rows.append(scope | {"seed": "macro", "seed_count": len(SEEDS), "status": "measured"}
                                        | {key: mean(r[key] for r in paired) for key in paired[0]})
    return rows


def score(rows, stability_by_case, decision, complete):
    output = {}
    for candidate, reference, label in CONTRASTS:
        paired = [r for r in rows if (r["candidate"], r["reference"], r["contrast"]) == (candidate, reference, label)
                  and (r["stage"], r["policy"], r["density_group"], r["window"]) == ("final", "ema", "all", "E36-E40")]
        by_seed = {r["seed"]: r for r in paired}
        key = f"{candidate}_vs_{reference}"
        if not complete or set(by_seed) != {*SEEDS, "macro"}:
            output[key] = {"decision": "unevaluable", "reason": "incomplete frozen screen"}
            continue
        macro = by_seed["macro"]
        gates = {"mae": macro["candidate_mae"] <= decision["mae_ratio_max"] * macro["reference_mae"],
                 "nAP": macro["delta_local_nap"] > decision["local_nap_delta_strict_min"]}
        if label in JOINT_GATES:
            gates["paired_directions"] = all(by_seed[seed]["delta_mae"] < 0 and by_seed[seed]["delta_local_nap"] > 0
                                             for seed in SEEDS)
            gates["stability"] = all(stability_by_case[seed, arm]["passed"] for seed in SEEDS for arm in (candidate, reference))
            for metric in ("rmse", "fp", "fn"):
                gates[metric] = macro[f"candidate_{metric}"] <= decision[f"{metric}_ratio_max"] * macro[f"reference_{metric}"]
        if label == "ungated_unshared_target":
            gates |= {"frozen_mae_target": macro["candidate_mae"] <= decision["ungated_unshared_mae_max"],
                      "frozen_nap_target": macro["candidate_local_nap"] > decision["ungated_unshared_local_nap_strict_min"]}
        output[key] = {"decision": "descriptive" if label == "state_increment" else "pass" if all(gates.values()) else "fail",
                       "checks": {} if label == "state_increment" else gates,
                       "ema_final_late_by_seed": by_seed,
                       "exploratory_nap_noninferiority": macro["delta_local_nap"] >= decision["exploratory_nap_margin"]}
    return output


def stagewise(seed, arm, readouts, images):
    grouped = defaultdict(dict)
    for row in readouts:
        grouped[row["epoch"], row["policy"], row["density_group"]][row["stage"]] = row
    image_pairs = defaultdict(dict)
    for row in images:
        key = row["epoch"], row["policy"], row["image_id"], row["density_group"], row["gt"]
        image_pairs[key][row["stage"]] = row
    changes, per_image = defaultdict(list), []
    for (epoch, policy, image, group, gt), pair in image_pairs.items():
        values = correction(pair["early"], pair["final"])
        for target in ("all", group):
            changes[epoch, policy, target].append(values)
        per_image.append({"seed": seed, "arm": arm, "epoch": epoch, "policy": policy, "image_id": image,
                          "density_group": group, "gt": gt, **values})
    rows = []
    for (epoch, policy, group), stages in grouped.items():
        early, final = stages["early"], stages["final"]
        deltas = {f"delta_{key}": final[key] - early[key] for key in METRICS}
        paired = changes[epoch, policy, group]
        if abs(mean(row["correction_abs_error"] for row in paired) - deltas["delta_mae"]) > 1e-12:
            raise ValueError("per-image correction differs from grouped MAE")
        rows.append({"seed": seed, "arm": arm, "epoch": epoch, "policy": policy, "density_group": group,
                     **deltas, "images_improved": sum(r["correction_abs_error"] < 0 for r in paired),
                     "images_tied": sum(r["correction_abs_error"] == 0 for r in paired),
                     "images_worsened": sum(r["correction_abs_error"] > 0 for r in paired),
                     "delta_canceled_unmatched": sum(r["delta_canceled_unmatched"] for r in paired)})
    return rows, per_image


def archive(root, output):
    config = read_json(CONFIG)
    owner, progress, phases, execution = close_out(root, config)
    archive_path = Path(config["historical_archive"])
    tables = read_frozen_files(archive_path)
    load_readouts(tables, archive_path)
    archived_readouts = {(seed, arm): metrics for (seed, arm, kind), metrics in tables.items() if kind == "metrics"}
    complete = progress["execution"] == "completed"
    done = set(progress["completed"])
    inputs = root / "run" / "preflight_cpu.json"
    preflight = read_json(inputs) if inputs.is_file() else None
    if preflight is not None and (preflight["status"] != "cpu_audit_pass" or preflight["config_sha256"] != file_sha(CONFIG)):
        raise ValueError("runtime cache and reference audit differs")
    elif complete and preflight is None:
        raise ValueError("completed queue is missing verified runtime inputs")
    readout = dict(archived_readouts)
    stability_by_case = {key: stability(tables[(*key, "training")], rows, config["decision"])
                         for key, rows in readout.items()}
    index, gate_stats, steps, per_image, evidence, integrity_errors = [], [], [], [], [], []
    for seed_text, order in config["order"].items():
        seed = int(seed_text)
        for arm in order:
            reference = config["variants"][arm]["reference"]
            case = f"{seed}/{arm}"
            try:
                detail, training, metrics, images, gates = verify_case(
                    root / "run" / "train" / seed_text / arm, seed, arm, config, preflight,
                    tables[seed, reference, "training"], case in done, owner["code_commit"],
                )
                ref_annot = {row["image_id"]: (int(row["gt"]), row["density_group"])
                             for row in tables[seed, reference, "images"] if row["epoch"] == "1"}
                if any((row["gt"], row["density_group"]) != ref_annot[row["image_id"]] for row in images):
                    raise ValueError("per-image annotations or density groups differ")
            except (ValueError, KeyError, FileNotFoundError) as error:
                integrity_errors.append({"seed": seed, "arm": arm, "error": str(error)})
                evidence.append({"seed": seed, "arm": arm, "status": "integrity_error"})
                index.append({"seed": seed, "arm": arm, "reference": reference, "status": "integrity_error",
                              "epochs": 0, "kind": "history", "path": "", "rows": 0, "bytes": 0, "sha256": ""})
                continue
            evidence.append(detail)
            status = detail["status"]
            head = {"seed": seed, "arm": arm, "reference": reference, "status": status, "epochs": detail["epochs"]}
            if not training:
                index.append(head | {"kind": "history", "path": "", "rows": 0, "bytes": 0, "sha256": ""})
                continue
            for label, rows in (("training", training), ("metrics", metrics), ("images", images)):
                relative = Path("epochs") / seed_text / arm / f"{label}.csv"
                (output / relative.parent).mkdir(parents=True, exist_ok=True)
                index.append(head | {"kind": label} | tracked_csv(output, relative, rows))
            gate_stats.extend(gates)
            grouped, paired = stagewise(seed, arm, metrics, images)
            steps.extend(grouped)
            per_image.extend(paired)
            if case in done:
                key = seed, arm
                readout[key] = metrics
                stability_by_case[key] = stability(training, metrics, config["decision"])
    complete &= not integrity_errors and len(readout) == len(archived_readouts) + config["limits_if_approved"]["runs"]
    write_csv(output / "epoch_index.csv", index)
    contrasts = paired_rows(readout, config["decision"])
    for name, rows in (("paired_contrasts", contrasts), ("gate_statistics", gate_stats),
                       ("stagewise_correction", steps), ("image_stagewise_correction", per_image)):
        if rows:
            write_csv(output / f"{name}.csv", rows)
    decisions = score(contrasts, stability_by_case, config["decision"], complete)
    qualifies = [arm for arm, key in (("RS", f"RS_vs_{SHARED}"), ("R", f"R_vs_{SHARED}"))
                 if decisions[key]["decision"] == "pass"]
    result = {
        "status": ("GPU_SCREEN_COMPLETE / READY_FOR_REVIEW" if complete else "BLOCKED / PARTIAL_EVIDENCE_ARCHIVED"),
        "evaluation": "completed" if complete else "unevaluable",
        "code_and_preregistration_commit": owner["code_commit"], "gpu_authorization": owner,
        "execution": execution | progress, "completed_fresh_runs": len(done), "archive_integrity_errors": integrity_errors,
        "planned_fresh_runs": config["limits_if_approved"]["runs"],
        "successful_project_updates": progress.get("successful_updates", 0),
        "planned_successful_updates": config["limits_if_approved"]["total_updates"],
        "historical_reference_reuse": "verified" if complete else "unevaluable",
        "criteria": decisions, "stability": {f"{seed}/{arm}": row for (seed, arm), row in stability_by_case.items()},
        "candidate_identity_transitions": "not_available", "qualifying_shared_arms": qualifies,
        "deployment_decision": "retain_existing_shared_2_aux_ema" if not qualifies else "hold_pending_replication",
        "replication": "not_authorized",
    }
    write_json(output / "research_decision.json", result)
    write_json(output / "run_evidence.json", {"execution": execution | progress, "cases": evidence,
                                              "resource_ledger_rows": len(phases)})
    write_json(output / "authorization.json", owner)
    for name in ("gpu_phases.jsonl", "synthetic_gpu_latency.json", "background_start.txt", "background_end.txt"):
        path = root / "run" / name
        if path.is_file():
            copyfile(path, output / name)
    print(json.dumps({"status": result["status"], "completed_fresh_runs": len(done),
                      "successful_project_updates": result["successful_project_updates"],
                      "qualifying_shared_arms": qualifies}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/LC-007D"))
    args = parser.parse_args()
    archive(args.run_root, args.output)


if __name__ == "__main__":
    main()
