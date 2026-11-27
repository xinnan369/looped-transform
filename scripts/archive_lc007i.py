"""Verify completed or partial native-crop evidence and score the fixed single-seed gate."""

import argparse
from pathlib import Path

from loopcrowd.experiment import file_sha, write_json
from scripts.archive_lc007b import GROUPS, local_ref, tracked_csv
from scripts.archive_lc007c import POLICIES
from scripts.archive_lc007d import close_out, endpoint, stability, verify_case
from scripts.archive_lc007e import compact_state_rows
from scripts.archive_lc007h_training import raw_case_refs
from scripts.prepare_lc007i import ARM, REF, SCREEN, SEED, reference_tables
from scripts.run_lc007a import write_csv
from scripts.run_lc007d import git, read_json
from scripts.run_lc007h_training import require_training_authorization
from scripts.run_lc007i import CASE, EXPOSURE_FIELDS


def score(training, metrics, historical, rule):
    epochs = rule["analysis_epochs"]
    a, b = [endpoint(rows, epochs, "ema", "final", "all") for rows in (metrics, historical)]
    stable = stability(training, metrics, rule)
    checks = {"mae": a["mae"] <= rule["mae_max"], "nap": a["local_nap"] > rule["nap_reference"],
              "stability": stable["passed"]}
    for k in ("rmse", "fp", "fn"):
        checks[k] = a[k] <= rule[f"{k}_ratio"] * b[k]
    return {"decision": "pass" if all(checks.values()) else "fail", "checks": checks, "candidate": a,
            "historical_512": b, "stability": stable}


def archive(root, output, plan):
    owner, progress, phases, runtime = close_out(root, plan, config_path=SCREEN, allow_truncated_tail=True)
    rev = owner["code_commit"]
    tree = git("rev-parse", f"{rev}^{{tree}}")
    require_training_authorization(owner, plan, file_sha(SCREEN), rev, tree)
    if not owner.get("independent_review") or progress.get("source_tree", tree) != tree:
        raise ValueError("execution freeze or review differs")
    historical = reference_tables(plan)
    run = root / "run"
    case = run / "train" / CASE
    evidence = {"raw_files": raw_case_refs(case)}
    complete = progress["execution"] == "completed" and progress["completed"] == [CASE]
    decision = {"decision": "unevaluable"}
    output.mkdir(parents=True)
    index = []
    try:
        if (case / "history.json").is_file():
            preflight = read_json(run / "preflight_cpu.json")
            if preflight["plan_sha256"] != file_sha(SCREEN):
                raise ValueError("runtime preflight config digest differs")
            rows = [row for row in preflight["exposure"] if row["crop_size"] == plan["train_crop_size"]]
            verified, training, metrics, images, _ = verify_case(
                case, SEED, ARM, plan, preflight, rows, complete, rev, depth_calls=4)
            evidence.update(verified)
            for expected, actual in zip(rows[:len(training)], training, strict=True):
                if any(expected[k] != actual[k] for k in EXPOSURE_FIELDS):
                    raise ValueError("archived exposure differs from the frozen trajectory")
            labels = {row["image_id"]: (int(row["gt"]), row["density_group"])
                      for row in historical["images"] if row["epoch"] == "1"}
            for row in images:
                if (row["gt"], row["density_group"]) != labels[row["image_id"]]:
                    raise ValueError("validation GT or density differs from the historical reference")
                row["canceled_unmatched"] = 2 * min(row["local_fp"], row["local_fn"])
            for name, values in (("training", training), ("metrics", metrics), ("images", images)):
                index.append({"kind": name} | tracked_csv(output, Path(f"{name}.csv"), values))
            corrections = list(compact_state_rows(metrics, images))
            if corrections:
                write_csv(output / "stagewise_correction.csv", corrections)
            if complete and verified["status"] == "completed":
                decision = score(training, metrics, historical["metrics"], plan["decision"])
                endpoints = []
                for (name, epochs) in (("E36-40", plan["decision"]["analysis_epochs"]), ("E40", [40])):
                    for policy in POLICIES:
                        for group in GROUPS:
                            for arm, table in ((ARM, metrics), (REF, historical["metrics"])):
                                endpoints.append({"arm": arm, "window": name, "policy": policy, "density_group": group}
                                                 | endpoint(table, epochs, policy, "final", group))
                write_csv(output / "endpoints.csv", endpoints)
        elif complete:
            raise ValueError("completed training lacks a history")
    except (KeyError, ValueError, OSError) as error:
        evidence.update(status="integrity_error", reason=str(error))
        decision = {"decision": "unevaluable"}
    if index:
        write_csv(output / "epoch_index.csv", index)
    write_json(output / "authorization.json", owner)
    observed = [run / file for file in ("preflight_cpu.json", "device_memory_start.json", "device_memory_end.json",
                "background_start.txt", "background_end.txt", "gpu_phases.jsonl", "queue_progress.json")]
    write_json(output / "run_evidence.json", {"execution": runtime | progress, "case": evidence, "phases": phases,
               "snapshots": [local_ref(path) for path in observed if path.is_file()]})
    write_json(output / "research_decision.json", decision | {
        "status": "COMPLETE / READY_FOR_REVIEW" if decision["decision"] != "unevaluable" else "BLOCKED / PARTIAL_EVIDENCE_ARCHIVED",
        "formal_baseline": "shared_2_aux_025", "historical_comparator": "non_contemporaneous_same_seed",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive(args.run_root, args.output, read_json(SCREEN))


if __name__ == "__main__":
    main()
