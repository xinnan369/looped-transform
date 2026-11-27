"""Audit completed or interrupted LC-007H training artifacts without a GPU."""

import argparse
from math import isfinite
from pathlib import Path

from loopcrowd.experiment import file_sha, write_json
from scripts.archive_lc007b import GROUPS, local_ref, tracked_csv
from scripts.archive_lc007c import POLICIES, SHARED, UNSHARED
from scripts.archive_lc007d import close_out, stability, verify_case
from scripts.archive_lc007e import endpoint_contrasts
from scripts.diagnose_lc007h import G, archives
from scripts.run_lc007a import read_json, write_csv
from scripts.run_lc007h_training import SCREEN, require_training_authorization, verify_inputs
from scripts.run_lc007d import git

JOINT = (("S-guide", "S4-shift", "shared_baseline"), ("U-guide", "U4-shift", "unshared_baseline"),
         ("S-guide", "S-placebo", "spatial_specificity"), ("S-guide", "U-guide", "matched_sharing"))
CONTRASTS = JOINT + tuple(("S-guide", ref, "historical_descriptive")
                          for ref in (SHARED, UNSHARED, "U4", "U4-shift", "cnn_only", "shared_1"))


def gain_rows(history, seed, arm):
    for epoch in history:
        for policy, key in (("live", "depth_live_validation"), ("ema", "ema_validation")):
            stats = epoch[key]["final"]["refinement_gain"]
            if set(stats) != set(GROUPS):
                raise ValueError("refinement gain density coverage differs")
            totals = [0, 0]
            for group in GROUPS:
                row = stats[group]
                if row is None:
                    if group == "all" or epoch[key]["final"]["density_groups"][group]["images"]:
                        raise ValueError("nonempty gain group has no token observations")
                    yield {"seed": seed, "arm": arm, "epoch": epoch["epoch"], "policy": policy,
                           "density_group": group, "valid_tokens": 0, "changed_tokens": 0,
                           **dict.fromkeys(("min", "max", "mean", "sd", "changed_fraction", "aligned_permuted_pearson"))}
                    continue
                cells, changed = row["valid_tokens"], row["changed_tokens"]
                correlation = row["aligned_permuted_pearson"]
                numeric = (row["min"], row["max"], row["mean"], row["sd"], row["changed_fraction"])
                if (not all(isfinite(value) for value in numeric)
                        or not .75 <= row["min"] <= row["mean"] <= row["max"] <= 1.25
                        or abs(row["mean"] - 1.) > 1e-6 or row["sd"] < 0
                        or type(cells) is not int or type(changed) is not int or not 0 <= changed <= cells or not cells
                        or row["changed_fraction"] != changed / cells
                        or (correlation is None) != (row["sd"] == 0)
                        or (correlation is not None and not -1 <= correlation <= 1)):
                    raise ValueError("invalid refinement gain moments")
                if group != "all":
                    totals[0] += cells
                    totals[1] += changed
                yield {"seed": seed, "arm": arm, "epoch": epoch["epoch"], "policy": policy,
                       "density_group": group, **row}
            if totals != [stats["all"]["valid_tokens"], stats["all"]["changed_tokens"]]:
                raise ValueError("gain denominators differ between aggregate and density strata")


def decisions(readouts, stable, rule, complete):
    paired, gates = endpoint_contrasts(readouts, stable, rule, complete, contrasts=CONTRASTS, joint=JOINT)
    for key, (mae, nap) in rule["primary_absolute_targets"].items():
        gate = gates[key]
        if gate["decision"] != "unevaluable":
            macro = gate["ema_final_late_by_seed"]["macro"]
            gate["checks"].update(absolute_mae=macro["candidate_mae"] <= mae, absolute_nap=macro["candidate_local_nap"] > nap)
            gate["decision"] = "pass" if all(gate["checks"].values()) else "fail"
    history = {}
    for alias, arm in (("S2", SHARED), ("U2", UNSHARED), ("U4-shift", "U4-shift")):
        macro = gates[f"S-guide_vs_{arm}"]["ema_final_late_by_seed"].get("macro")
        mae, nap = rule["historical_absolute_targets"][alias]
        if complete and macro:
            history[alias] = {"mae": macro["candidate_mae"], "nap": macro["candidate_local_nap"],
                              "mae_pass": macro["candidate_mae"] <= mae, "nap_pass": macro["candidate_local_nap"] > nap}
            if alias == "S2":
                history[alias]["parity_mae_pass"] = macro["candidate_mae"] <= rule["historical_absolute_targets"]["S2_parity_mae_max"]
    return paired, gates, history


def raw_case_refs(path):
    names = ("run_manifest.json", "history.json", "checkpoint_index.json", "last.pt", "last.tmp",
             "interrupted.pt", "interrupted.tmp", "mismatched_epoch.json")
    files = [path / name for name in names] + sorted((path / "images").glob("*.json"))
    return [local_ref(item) for item in files if item.is_file()]


def archive(root, output, screen):
    verify_inputs(screen)
    owner, progress, phases, runtime = close_out(root, screen, config_path=SCREEN, allow_truncated_tail=True)
    revision = owner["code_commit"]
    tree = git("rev-parse", f"{revision}^{{tree}}")
    require_training_authorization(owner, screen, file_sha(SCREEN), revision, tree)
    if progress.get("source_tree", tree) != tree or progress.get("config_sha256", file_sha(SCREEN)) != file_sha(SCREEN):
        raise ValueError("execution source or config digest differs from authorization")
    if output.exists():
        raise ValueError("archive requires a new output directory")
    output.mkdir(parents=True)
    tables = archives()
    run = root / "run"
    path = run / "preflight_cpu.json"
    preflight = read_json(path) if path.is_file() else {}
    frozen = read_json(Path("reports/LC-007H/initialization_audit.json"))
    if preflight and (preflight["plan_sha256"] != file_sha(SCREEN)
                      or preflight["arm_config_sha256"] != file_sha(screen["arm_config"])
                      or preflight["data"]["manifest_sha256"] != frozen["data"]["manifest_sha256"]
                      or preflight["data"]["pretrained_weight_sha256"] != frozen["data"]["pretrained_weight_sha256"]
                      or preflight["planned_starts"] != frozen["planned_initializations"]):
        raise ValueError("runtime preflight does not match frozen input and model identities")
    historical = read_json(G / "run_evidence.json")
    readouts = {(seed, arm): rows for (seed, arm, kind), rows in tables.items() if kind == "metrics"}
    rule = screen["decision"]
    stability_rule = {"analysis_epochs": rule["epochs"], "loss_reference_epochs": rule["loss_reference_epochs"],
                      "late_mae_sample_cv_max": rule["sample_cv_max"],
                      "absolute_relative_final_loss_change_max": rule["absolute_loss_relative_change_max"]}
    stable = {key: stability(tables[(*key, "training")], rows, stability_rule) for key, rows in readouts.items()}
    evidence, index, gains = [], [], []
    valid_cases = 0
    for seed_text, arms in screen["order"].items():
        for arm in arms:
            seed, label = int(seed_text), f"{seed_text}/{arm}"
            reference = screen["arms"][arm]["comparison"]
            case_path = run / "train" / label
            detail = {"seed": seed, "arm": arm, "reference": reference, "raw_files": raw_case_refs(case_path)}
            complete = label in progress["completed"]
            try:
                checked, training, metrics, images, _ = verify_case(case_path, seed, arm, screen, preflight,
                    tables[seed, reference, "training"], complete, revision, depth_calls=4)
                detail.update(checked)
                if complete and detail["status"] != "completed":
                    raise ValueError("listed complete case had no verified history")
                if training:
                    old = next(r["manifest"] for r in historical["cases"] if r["seed"] == seed and r["arm"] == reference)
                    old_path = Path(old["path"])
                    if file_sha(old_path) != old["sha256"]:
                        raise ValueError("historical run manifest digest differs")
                    run_meta = read_json(case_path / "run_manifest.json")
                    old_meta = read_json(old_path)
                    recipe = {key: value for key, value in run_meta["config"].items() if key != "refinement"}
                    if (run_meta["eval_protocol"] != old_meta["eval_protocol"] or run_meta["ema"] != old_meta["ema"]
                            or recipe != old_meta["config"]):
                        raise ValueError("reference training or metric semantics differ")
                    summaries = list(gain_rows(read_json(case_path / "history.json"), seed, arm))
                    labels = {row["image_id"]: (int(row["gt"]), row["density_group"])
                              for row in tables[seed, reference, "images"] if row["epoch"] == "1"}
                    for row in images:
                        if (row["gt"], row["density_group"]) != labels[row["image_id"]]:
                            raise ValueError("validation GT or density strata differ from the reference")
                        row["canceled_unmatched"] = 2 * min(row["local_fp"], row["local_fn"])
                        if row["canceled_unmatched"] != row["local_fp"] + row["local_fn"] - row["abs_count_error"]:
                            raise ValueError("image count cancellation differs")
                    for kind, rows in (("training", training), ("metrics", metrics), ("images", images)):
                        rel = Path("epochs") / seed_text / arm / f"{kind}.csv"
                        index.append({"seed": seed, "arm": arm, "status": detail["status"], "kind": kind}
                                     | tracked_csv(output, rel, rows))
                    gains.extend(summaries)
                    if complete:
                        readouts[seed, arm], stable[seed, arm] = metrics, stability(training, metrics, stability_rule)
                        valid_cases += 1
            except (ValueError, KeyError, TypeError, OSError) as exc:
                detail.update(status="integrity_error", reason=str(exc))
            evidence.append(detail)
    complete = progress["execution"] == "completed" and valid_cases == screen["limits_if_approved"]["runs"]
    paired, gates, targets = decisions(readouts, stable, rule, complete)
    for name, rows in (("epoch_index", index), ("refinement_gain", gains), ("paired_contrasts", paired)):
        if rows:
            write_csv(output / f"{name}.csv", rows)
    write_json(output / "authorization.json", owner)
    resource_files = [run / name for name in ("queue_progress.json", "gpu_phases.jsonl", "preflight_cpu.json",
                      "background_start.txt", "background_end.txt", "device_memory_start.json", "device_memory_end.json")]
    write_json(output / "run_evidence.json", {"execution": runtime | progress, "cases": evidence, "phases": phases,
               "resource_files": [local_ref(path) for path in resource_files if path.is_file()]})
    write_json(output / "research_decision.json", {
        "status": "COMPLETE / READY_FOR_REVIEW" if complete else "BLOCKED / PARTIAL_EVIDENCE_ARCHIVED",
        "verified_runs": valid_cases, "gates": gates, "historical_absolute_targets": targets,
        "formal_baseline": SHARED, "historical_controls": "non_contemporaneous",
        "identity_diagnostics": "not_run_optional", "next_action": "review",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.run_root, args.output, read_json(SCREEN))


if __name__ == "__main__":
    main()
