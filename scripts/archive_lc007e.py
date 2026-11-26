"""Verify and score the frozen four-call screen from its local CPU-readable artifacts."""

import argparse
from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path
from shutil import copyfile
from statistics import mean

from loopcrowd.experiment import file_sha, write_json
from scripts.archive_lc007b import GROUPS, METRICS, tracked_csv
from scripts.archive_lc007c import POLICIES, SEEDS, SHARED, UNSHARED, correction
from scripts.archive_lc007d import close_out, endpoint, stability, verify_case
from scripts.prepare_lc007e import SCREEN, history
from scripts.run_lc007a import read_json, write_csv

STAGES = ("stage1", "early", "stage3", "final")
JOINT = (("S4", SHARED, "depth_ungated"), ("S4RS", "RS", "depth_RS"),
         ("S4", "U4", "sharing_ungated"), ("S4RS", "U4RS", "sharing_RS"))
CONTRASTS = JOINT + (
    ("U4", UNSHARED, "depth_ungated_unshared"), ("U4RS", "U-RS", "depth_RS_unshared"),
    ("S4RS", "S4", "gating_shared"), ("U4RS", "U4", "gating_unshared"),
    *((arm, ref, "historical_target") for arm in ("S4", "S4RS") for ref in ("cnn_only", "shared_1", UNSHARED)),
    *((arm, "R", "historical_descriptive") for arm in ("S4", "S4RS")),
)
INTERACTIONS = (("S4", "U4", SHARED, UNSHARED, "ungated"),
                ("S4RS", "U4RS", "RS", "U-RS", "RS"))
SELECTION = (("S4", (SHARED, "U4")), ("S4RS", ("RS", "U4RS")))


def compact_state_rows(readouts, images, *, stage_names=STAGES, first_full_epoch=1, midpoint_final=False):
    observed, image_groups = defaultdict(dict), defaultdict(dict)
    for row in readouts:
        if row["density_group"] == "all":
            if (row["nonfinite"] != 0 or any(not isfinite(row[key]) or row[key] < 0
                                             for key in ("state_rms", "update_rms"))):
                raise ValueError("invalid depth state summary")
            expected = row["update_rms"] / row["state_rms"] if row["state_rms"] else None
            if row["relative_rms"] != expected:
                raise ValueError("inconsistent relative depth state RMS")
        observed[row["epoch"], row["policy"], row["density_group"]][row["stage"]] = row
    for row in images:
        for group in ("all", row["density_group"]):
            image_groups[row["epoch"], row["policy"], group].setdefault(row["image_id"], {})[row["stage"]] = row
    for (epoch, policy, group), stages in observed.items():
        image_stages = image_groups[epoch, policy, group].values()
        names = stage_names if epoch >= first_full_epoch else ("early", "final")
        comparisons = list(zip(names, names[1:])) + [(names[0], names[-1])]
        if midpoint_final:
            comparisons.append(("early", "final"))
        for low, high in dict.fromkeys(comparisons):
            changes = [correction(image[low], image[high]) for image in image_stages]
            deltas = {f"delta_{metric}": stages[high][metric] - stages[low][metric] for metric in METRICS}
            if not isclose(mean(row["correction_abs_error"] for row in changes), deltas["delta_mae"], abs_tol=1e-12):
                raise ValueError("grouped and image-level stage corrections differ")
            yield {"epoch": epoch, "policy": policy, "density_group": group, "before": low, "after": high,
                   **deltas, "images_improved": sum(row["correction_abs_error"] < 0 for row in changes),
                   "images_tied": sum(row["correction_abs_error"] == 0 for row in changes),
                   "images_worsened": sum(row["correction_abs_error"] > 0 for row in changes),
                   "delta_canceled_unmatched": sum(row["delta_canceled_unmatched"] for row in changes)}


def endpoint_contrasts(readouts, stable, decision, complete, *, contrasts=CONTRASTS, joint=JOINT):
    results, checks = [], {}
    for candidate, reference, label in contrasts:
        primary = {}
        for window, epochs in (("E36-E40", decision["epochs"]), ("E40", (40,))):
            for policy in POLICIES:
                for group in GROUPS:
                    scope = {"candidate": candidate, "reference": reference, "comparison": label, "window": window,
                             "policy": policy, "stage": "final", "density_group": group, "status": "measured"}
                    values = {}
                    for seed in SEEDS:
                        if (seed, candidate) not in readouts or (seed, reference) not in readouts:
                            continue
                        after, before = [endpoint(readouts[seed, arm], epochs, policy, "final", group)
                                         for arm in (candidate, reference)]
                        values[seed] = {f"candidate_{key}": after[key] for key in METRICS} | {
                            f"reference_{key}": before[key] for key in METRICS} | {
                            f"delta_{key}": after[key] - before[key] for key in METRICS}
                    if len(values) == len(SEEDS):
                        values["macro"] = {key: mean(row[key] for row in values.values()) for key in values[SEEDS[0]]}
                    results.extend(scope | {"seed": seed} | row for seed, row in values.items())
                    if (window, policy, group) == ("E36-E40", "ema", "all"):
                        primary = values
        key = f"{candidate}_vs_{reference}"
        outcome = {"comparison": label, "decision": "unevaluable", "ema_final_late_by_seed": primary}
        if complete and "macro" in primary:
            macro = primary["macro"]
            passed = {"mae": macro["candidate_mae"] <= decision["mae_ratio_max"] * macro["reference_mae"],
                      "nap": macro["delta_local_nap"] > decision["nap_strict_delta_min"]}
            descriptive = (candidate, reference, label) not in joint and label != "historical_target"
            if (candidate, reference, label) in joint:
                passed["paired_directions"] = all(primary[seed]["delta_mae"] < 0 and
                                                   primary[seed]["delta_local_nap"] > 0 for seed in SEEDS)
                passed["stability"] = all(stable[seed, arm]["passed"] for seed in SEEDS for arm in (candidate, reference))
                for metric in ("rmse", "fp", "fn"):
                    passed[metric] = macro[f"candidate_{metric}"] <= decision[f"{metric}_ratio_max"] * macro[f"reference_{metric}"]
            outcome.update(decision="descriptive" if descriptive else "pass" if all(passed.values()) else "fail",
                           checks={} if descriptive else passed)
        checks[key] = outcome
    return results, checks


def interactions(readouts, decision, *, strata=INTERACTIONS):
    for shared4, unshared4, shared2, unshared2, stratum in strata:
        arms = (shared4, unshared4, shared2, unshared2)
        for window, epochs in (("E36-E40", decision["epochs"]), ("E40", (40,))):
            for policy in POLICIES:
                paired = {}
                for seed in SEEDS:
                    if not all((seed, arm) in readouts for arm in arms):
                        continue
                    a, b, c, d = [endpoint(readouts[seed, arm], epochs, policy, "final", "all") for arm in arms]
                    paired[seed] = {f"delta_{metric}": (a[metric] - b[metric]) - (c[metric] - d[metric]) for metric in METRICS}
                if len(paired) == len(SEEDS):
                    paired["macro"] = {key: mean(row[key] for row in paired.values()) for key in paired[SEEDS[0]]}
                for seed, row in paired.items():
                    yield {"candidate": f"{shared4}-{unshared4}", "reference": f"{shared2}-{unshared2}",
                           "comparison": f"sharing_gap_interaction_{stratum}", "window": window, "policy": policy,
                           "stage": "final", "density_group": "all", "status": "measured", "seed": seed, **row}


def archive(root, output, *, screen_path=SCREEN, history_tables=history, contrasts=CONTRASTS, joint=JOINT,
            strata=INTERACTIONS, selection=SELECTION, negative_protocol="separate_K8_proposal", depth_calls=4):
    screen = read_json(screen_path)
    owner, progress, phases, runtime = close_out(root, screen, config_path=screen_path)
    tables = history_tables()
    run = root / "run"
    preflight = read_json(run / "preflight_cpu.json") if (run / "preflight_cpu.json").is_file() else {}
    if preflight and (preflight["plan_sha256"] != file_sha(screen_path)
                      or preflight["arm_config_sha256"] != file_sha(Path(screen["arm_config"]))):
        raise ValueError("runtime preflight digests differ from the frozen plans")
    first_full_epoch = read_json(Path(screen["arm_config"])).get("depth_readout_start_epoch", 1)
    labels_by_call = {depth_calls // 2: "early", depth_calls: "final"}
    stage_names = tuple(labels_by_call.get(call, f"stage{call}") for call in range(1, depth_calls + 1))
    readouts = {(seed, arm): rows for (seed, arm, kind), rows in tables.items() if kind == "metrics"}
    rule = screen["decision"]
    stability_rule = {"analysis_epochs": rule["epochs"], "loss_reference_epochs": rule["loss_reference_epochs"],
                      "late_mae_sample_cv_max": rule["sample_cv_max"],
                      "absolute_relative_final_loss_change_max": rule["absolute_loss_relative_change_max"]}
    stable = {key: stability(tables[(*key, "training")], rows, stability_rule) for key, rows in readouts.items()}
    done = set(progress["completed"])
    index, stats, corrections, evidence, errors = [], [], [], [], []
    for seed_text, arms in screen["order"].items():
        seed = int(seed_text)
        for arm in arms:
            reference = screen["arms"][arm]["comparison"]
            case = f"{seed}/{arm}"
            base = {"seed": seed, "arm": arm, "reference": reference}
            try:
                detail, training, metrics, images, gates = verify_case(
                    run / "train" / seed_text / arm, seed, arm, screen, preflight, tables[seed, reference, "training"],
                    case in done, owner["code_commit"], depth_calls=depth_calls)
                labels = {row["image_id"]: (int(row["gt"]), row["density_group"])
                          for row in tables[seed, reference, "images"] if row["epoch"] == "1"}
                if any((row["gt"], row["density_group"]) != labels[row["image_id"]] for row in images):
                    raise ValueError("validation image labels or density groups differ from the archive")
                case_corrections = list(compact_state_rows(metrics, images, stage_names=stage_names,
                                                           first_full_epoch=first_full_epoch,
                                                           midpoint_final=first_full_epoch > 1))
            except (ValueError, KeyError, OSError) as error:
                detail = base | {"status": "integrity_error", "epochs": 0, "error": str(error)}
                errors.append(detail)
                training = []
            evidence.append(detail)
            base |= {"status": detail["status"], "epochs": detail["epochs"]}
            if not training:
                index.append(base | {"kind": "history", "path": "", "rows": 0, "bytes": 0, "sha256": ""})
                continue
            for kind, rows in (("training", training), ("metrics", metrics), ("images", images)):
                relative = Path("epochs") / seed_text / arm / f"{kind}.csv"
                index.append(base | {"kind": kind} | tracked_csv(output, relative, rows))
            stats.extend(gates)
            corrections.extend({"seed": seed, "arm": arm, **row} for row in case_corrections)
            if case in done:
                readouts[seed, arm] = metrics
                stable[seed, arm] = stability(training, metrics, stability_rule)
    verified = sum(detail["status"] == "completed" and detail["epochs"] == 40 for detail in evidence)
    complete = progress["execution"] == "completed" and not errors and verified == screen["limits_if_approved"]["runs"]
    paired, checks = endpoint_contrasts(readouts, stable, rule, complete, contrasts=contrasts, joint=joint)
    interaction = list(interactions(readouts, rule, strata=strata))
    qualifies = [arm for arm, references in selection if all(checks[f"{arm}_vs_{reference}"]["decision"] == "pass"
                                                              for reference in references)]
    write_csv(output / "epoch_index.csv", index)
    for name, rows in (("gate_statistics", stats), ("paired_contrasts", paired + interaction),
                       ("stagewise_correction", corrections)):
        if rows:
            write_csv(output / f"{name}.csv", rows)
    summary = {
        "status": "GPU_SCREEN_COMPLETE / READY_FOR_REVIEW" if complete else "BLOCKED / PARTIAL_EVIDENCE_ARCHIVED",
        "evaluation": "completed" if complete else "unevaluable", "authorization": owner,
        "execution": runtime | progress, "verified_runs": verified, "archive_integrity_errors": errors,
        "criteria": checks, "stability": {f"{seed}/{arm}": item for (seed, arm), item in stable.items()},
        "interactions": [row for row in interaction if row["seed"] == "macro" and row["policy"] == "ema"
                         and row["window"] == "E36-E40"],
        "qualifying_arms": qualifies, "candidate_identity_transitions": "not_available",
        "next_protocol": ("separate_A_B_replication" if qualifies else negative_protocol) if complete else "review_stop",
    }
    if first_full_epoch > 1:
        summary["readout_schedule"] = {"epochs_before_full": [1, first_full_epoch - 1], "reads": ["early", "final"],
                                       "other_calls": "not_run", "full_start_epoch": first_full_epoch}
    write_json(output / "research_decision.json", summary)
    write_json(output / "run_evidence.json", {"execution": runtime | progress, "cases": evidence, "phase_rows": len(phases)})
    write_json(output / "authorization.json", owner)
    for filename in ("gpu_phases.jsonl", "background_start.txt", "background_end.txt", "synthetic_gpu_latency.json",
                     "device_memory_start.json", "device_memory_end.json"):
        if (run / filename).is_file():
            copyfile(run / filename, output / filename)
    print(summary["status"], verified, "verified runs", progress.get("successful_updates", 0), "updates", qualifies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive(args.run_root, args.output)


if __name__ == "__main__":
    main()
