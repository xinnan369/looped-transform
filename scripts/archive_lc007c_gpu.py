"""Validate and publish compact evidence from the frozen CPU-to-GPU endpoint run."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from loopcrowd.experiment import file_sha, write_json
from scripts.prepare_lc007c import PILOT_CAP
from scripts.run_lc007a import read_json, write_csv
from scripts.run_lc007c import POLICIES, PROBE, PROBE_SHA

OUTPUT = Path("reports/LC-007C")
FIELDS = ("cosine", "left_norm", "right_norm", "weighted_left_right_norm_ratio",
          "weighted_left_projection_on_right", "right_projection_on_weighted_left")


def validate_cases(run, entries, plan):
    batches = {row["seed"]: row["batches"] for row in plan["plans"]}
    expected = [(ref["seed"], ref["arm"], b["index"], p, b["density_group"], b["batch_sha256"])
                for ref in plan["checkpoints"] for b in batches[ref["seed"]] for p in POLICIES]
    keys = ("seed", "arm", "batch", "policy", "density_group", "batch_sha256")
    observed = [tuple(row[key] for key in keys) for row in entries]
    if observed != expected[:len(observed)]:
        raise ValueError("executed order is not the prefix of the frozen case matrix")
    cases = [row for row in entries if row["status"] == "completed"]
    if len(cases) != run["completed_cases"] or len(cases) > run.get("entered_cases", 0):
        raise ValueError("case accounting differs from the execution ledger")
    hashes = {}
    for case in cases:
        identity = case["seed"], case["batch"]
        if hashes.setdefault(identity, case["batch_tensor_sha256"]) != case["batch_tensor_sha256"]:
            raise ValueError("transformed batch digest differs across project reads")
        comparisons = case["measurement"]["comparisons"]
        expected_rows = 6 if case["arm"] == "shared_2_aux_025" else 7
        if len(comparisons) != expected_rows or any(row["nonfinite"] for row in comparisons):
            raise ValueError("incomplete or nonfinite gradients in a completed case")
    if run["status"] == "completed":
        if len(cases) != 96 or len(entries) != 96 or len(hashes) != 24 or run["entered_cases"] != 96:
            raise ValueError("completed execution must have the exact 96 cases")
        if run["gpu_seconds"] > PILOT_CAP["gpu_seconds"] or run["peak_reserved_bytes"] > PILOT_CAP["process_peak_reserved_bytes"]:
            raise ValueError("completed execution exceeds the resource cap")
        if len(run["checkpoints"]) != 4 or any(not row["checkpoint_unchanged"] or
                                               not all(row["invariance"].values())
                                               for row in run["checkpoints"].values()):
            raise ValueError("completed execution lacks checkpoint invariance evidence")
        if any(p["budget_crossings"] or p["final_sync_error"] for p in run["phases"]):
            raise ValueError("completed execution has a resource or synchronization violation")
    if run.get("project_optimizer_updates") != 0 or run.get("external_experiment_download_bytes") != 0:
        raise ValueError("unexpected project updates or downloads")
    return cases


def summaries(rows):
    groups = defaultdict(list)
    keys = ("seed", "arm", "policy", "density_group", "comparison", "group", "relation")
    for row in rows:
        for seed in (row["seed"], "macro"):
            for density in (row["density_group"], "all"):
                group = row | {"seed": seed, "density_group": density}
                groups[tuple(group[key] for key in keys)].append(row)
    result = []
    for key, items in groups.items():
        measured = [row for row in items if row["status"] == "measured"]
        conflict = sum(row["conflict"] is True for row in measured)
        row = dict(zip(keys, key)) | {"cases": len(items), "measured": len(measured),
                                     "no_gradient_path": sum(r["status"] == "no_gradient_path" for r in items),
                                     "near_zero": sum(r["status"] == "zero_or_near_zero_norm" for r in items),
                                     "conflict": conflict,
                                     "conflict_fraction": conflict / len(measured) if measured else None}
        for field in FIELDS:
            available = [r[field] for r in items if r[field] is not None]
            if available:
                for name, value in zip(("p10", "median", "p90"), np.quantile(available, (.1, .5, .9))):
                    row[f"{field}_{name}"] = float(value)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run = read_json(args.local / "execution.json")
    events = args.local / "cases.jsonl"
    entries = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()] if events.exists() else []
    if run["probe_manifest_sha256"] != PROBE_SHA or file_sha(PROBE) != PROBE_SHA:
        raise ValueError("approved probe manifest checksum differs")
    cases = validate_cases(run, entries, read_json(PROBE))
    rows, handling = [], []
    for case in cases:
        meta = {key: value for key, value in case.items() if key != "measurement"}
        measured = case["measurement"]
        rows.extend(meta | {"relation": "same_physical_parameter_basis"} | row for row in measured["comparisons"])
        handling.append(meta | {key: value for key, value in measured.items() if key != "comparisons"})
    failures = [case for case in entries if case["status"] != "completed"]
    if rows:
        write_csv(args.output / "gradient_diagnostics.csv", rows)
        write_csv(args.output / "gradient_summary.csv", summaries(rows))
    else:
        write_csv(args.output / "gradient_diagnostics.csv", [
            {"status": "not_run_" + run["status"], "actual_cases": 0, "reason": run.get("error"), "cosine": None},
        ])
    prior = read_json(args.output / "gradient_equivalence.json")
    prior.update(real_checkpoint_gradient_execution=run["status"], gpu_synthetic_preflight=run["synthetic_preflight"],
                 gpu_case_equivalence=handling)
    write_json(args.output / "gradient_equivalence.json", prior)
    run["retained_case_failures"] = failures
    run["local_evidence"] = {}
    for path in (args.local / "execution.json", events, args.local.parent / "loopcrowd-lc007c-gpu.log",
                 args.local.parent / "loopcrowd-lc007c-gpu-background.csv"):
        if path.exists():
            run["local_evidence"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha(path)}
    write_json(args.output / "gpu_execution.json", run)
    print(f"{run['status']}: {len(cases)} cases, {len(rows)} comparison rows, {run.get('gpu_seconds', 0):.3f}s")


if __name__ == "__main__":
    main()
