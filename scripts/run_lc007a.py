"""Check historical provenance, then replay the frozen LC-007A validation cases."""

import argparse
import csv
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from loopcrowd.accuracy_diagnosis import aggregate, analyze_image, density
from loopcrowd.calibration import COUNT_CUTOFF, DENSITY_GROUPS, SCORE_FLOOR, json_digest
from loopcrowd.data import load_rgb
from loopcrowd.diagnostics import backbone_bn_buffers, describe_bn
from loopcrowd.experiment import file_sha, load_checkpoint, make_model, write_json
from loopcrowd.inference import predict_image, predict_image_stages
from loopcrowd.manifest import read_manifest

ROOT = Path("reports")
BASE = "7848efc8dd572edb5de43b2f3aaa0e5eff46f209"
SEEDS = (42, 123, 2026)
LABEL_POLICY = "shanghaitech_lc001_drop_only_logged_out_of_bounds"
DATA = {
    "A": ("e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316", 270, 30, 14177),
    "B": ("de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482", 360, 40, 4295),
}
B_ARMS = ("cnn", "shared_1", "shared_2_final_only", "shared_2_aux_025",
          "unshared_2_final_only", "unshared_2_aux_025")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


@lru_cache(maxsize=None)
def local_digest(path):
    path = Path(path).expanduser()
    return file_sha(path) if path.is_file() else None


def local_status(artifact):
    actual = local_digest(artifact["path"])
    if actual is None:
        return "not_run_missing"
    return "verified" if actual == artifact["sha256"] else "blocked_sha_mismatch"


def checkpoint_columns(key, artifact):
    return {f"{key}_{name}": artifact[name] for name in ("path", "sha256")} | {
        f"{key}_local": local_status(artifact),
    }


def archive_runs():
    runs = {}
    for issue, index in (("LC-003", "training_history.json"), ("LC-005", "run_evidence.json")):
        path = ROOT / issue / index
        for run_id, run in read_json(path)["runs"].items():
            runs[f"{issue}/{run_id}"] = (run, path, "fresh_train")
    issue = "LC-004A"
    for run_id, ref in read_json(ROOT / issue / "training_history.json")["runs"].items():
        path = ROOT / issue / ref["path"]
        action = "training_record_reuse" if ref["source"] == "LC-003 reuse" else "fresh_train"
        runs[f"{issue}/{run_id}"] = (read_json(path), path, action)
    issue = "LC-006"
    for ref in read_csv(ROOT / issue / "training_history.csv"):
        path = ROOT / issue / ref["provenance_json"]
        runs[f"{issue}/{ref['seed']}/{ref['arm']}"] = (read_json(path), path, "fresh_train")
    return runs


def create_ledger(runs):
    rows, provenance_columns = [], {}
    for run_id, (run, source, action) in sorted(runs.items()):
        p = run["provenance"]
        data = p["data"]
        if json_digest(p["config"]) != p["config_sha256"]:
            raise ValueError(f"archived config SHA differs for {run_id}")
        history = run["raw_history"] if "raw_history" in run else run["local_history"]
        provenance_columns[run_id] = {
            "manifest_path": data["manifest"], "manifest_sha256": data["sha256"],
            "manifest_local": local_status({"path": data["manifest"], "sha256": data["sha256"]}),
            "label_policy": p["label_policy"], "config_sha256": p["config_sha256"],
            "training_code_sha": p["git_commit"], "initial_weight_sha256": p["weight_initialization"]["sha256"],
            "source": str(source), "source_sha256": file_sha(source),
        }
        reused_from = ""
        if action == "training_record_reuse":
            parts = run_id.split("/", 2)
            reused_from = "LC-003/warmup_cosine/" + parts[2]
            if p != runs[reused_from][0]["provenance"]:
                raise ValueError(f"training reuse provenance differs for {run_id}")
        rows.append({
            "record": run_id, "activity": action, "reused_from": reused_from, "recorded_epochs": p["completed_epochs"],
            "new_optimizer_updates": p["global_step"] if action == "fresh_train" else 0,
        } | provenance_columns[run_id] | checkpoint_columns("history", history)
            | checkpoint_columns("best", p["checkpoints"]["best.pt"])
            | checkpoint_columns("last", p["checkpoints"]["last.pt"]))
    legacy_path = ROOT / "LC-003/diagnosis.json"
    legacy = read_json(legacy_path)
    for arm, data in legacy["runs"].items():
        rows.append({
            "record": f"LC-003/LC-002R/42/{arm}/best", "activity": "checkpoint_replay",
            "reused_from": f"LC-002R/42/{arm}", "new_optimizer_updates": 0,
            "source": str(legacy_path), "source_sha256": file_sha(legacy_path),
            "manifest_path": rows[0]["manifest_path"], "manifest_sha256": DATA["A"][0],
            "manifest_local": rows[0]["manifest_local"], "label_policy": LABEL_POLICY,
            "training_code_sha": data["historical_code_sha"], "config_sha256": data["historical_config_sha"],
        } | checkpoint_columns("best", data["checkpoints"]["best.pt"])
            | checkpoint_columns("last", data["checkpoints"]["last.pt"]))
    index = read_json(ROOT / "LC-004A/loop_diagnosis.json")
    for field, action in (("checkpoint_diagnostics", "checkpoint_replay"),
                          ("frozen_bn_checkpoint_buffers", "metadata_only")):
        for case, ref in index[field].items():
            family, seed, arm, snapshot = case.split("/")
            source_run = f"LC-003/warmup_cosine/{seed}/{arm}" if family == "lc003" else f"LC-004A/{family}/{seed}/{arm}"
            checkpoint = runs[source_run][0]["provenance"]["checkpoints"][f"{snapshot}.pt"]
            if ref.get("checkpoint_sha256", checkpoint["sha256"]) != checkpoint["sha256"]:
                raise ValueError(f"replay checkpoint differs for {case}")
            source = ROOT / "LC-004A" / ref["path"]
            rows.append({
                "record": f"LC-004A/{case}", "activity": action, "reused_from": source_run,
                "new_optimizer_updates": 0, "snapshot_epoch": ref.get("epoch"),
            } | provenance_columns[source_run] | {"source": str(source), "source_sha256": file_sha(source)}
                | checkpoint_columns("evaluated", checkpoint))
    source = ROOT / "LC-004B/calibration_thresholds.csv"
    for case in read_csv(source):
        source_run = f"LC-003/warmup_cosine/{case['seed']}/{case['variant']}"
        checkpoint = runs[source_run][0]["provenance"]["checkpoints"]["last.pt"]
        p = provenance_columns[source_run]
        if any(case[key] != expected for key, expected in (
            ("checkpoint_sha256", checkpoint["sha256"]), ("config_sha256", p["config_sha256"]),
            ("training_code_sha", p["training_code_sha"]),
        )):
            raise ValueError(f"calibration provenance differs for {case['case']}")
        rows.append({
            "record": f"LC-004B/{case['case']}", "activity": "checkpoint_call_calibration_and_replay",
            "reused_from": source_run, "new_optimizer_updates": 0, "calls": case["calls"], "snapshot_epoch": 40,
        } | p | {"source": str(source), "source_sha256": file_sha(source)} | checkpoint_columns("evaluated", checkpoint))
    if sum(row["activity"] == "fresh_train" for row in rows) != 47:
        raise ValueError("historical fresh-train denominator differs from the frozen archive")
    return rows


def planned_cases():
    for seed in SEEDS:
        for arm, source_arm in (("cnn", "cnn_only"), ("shared_1", "shared_1"),
                                ("unshared_2_final_only", "unshared_2")):
            yield "A", seed, arm, f"LC-003/warmup_cosine/{seed}/{source_arm}", False
        for arm in ("final_only", "aux_025"):
            yield "A", seed, f"shared_2_{arm}", f"LC-005/{seed}/{arm}", True
        for arm in B_ARMS:
            yield "B", seed, arm, f"LC-006/{seed}/{arm}", arm in B_ARMS[2:]


def load_dataset(dataset, directory):
    sha, train_count, val_count, gt_count = DATA[dataset]
    manifest = directory / f"shanghaitech_{dataset.lower()}.jsonl"
    if local_digest(str(manifest)) != sha:
        raise ValueError("manifest missing or SHA mismatch")
    scope = read_manifest(manifest, splits=("train", "val"))
    train = [record for record in scope if record.split == "train"]
    validation = [record for record in scope if record.split == "val"]
    if (len(train), len(validation), sum(len(row.points) for row in validation)) != (train_count, val_count, gt_count):
        raise ValueError("manifest train/validation population differs")
    boundaries = np.quantile([density(row) for row in train], (1 / 3, 2 / 3)).tolist()
    return manifest, validation, boundaries


def aggregate_stages(rows, ranked):
    result = {"all": aggregate(rows, ranked)}
    for group in DENSITY_GROUPS:
        indices = [index for index, row in enumerate(rows) if row["density_group"] == group]
        result[group] = aggregate([rows[index] for index in indices], [ranked[index] for index in indices])
    top = sorted(rows, key=lambda row: (-row["absolute_error"], row["image_id"]))[:5]
    return result | {
        "largest_absolute_errors": [{"image_id": row["image_id"], "absolute_error": row["absolute_error"],
                                     "density_group": row["density_group"]} for row in top],
        "absolute_error_sum": sum(row["absolute_error"] for row in rows),
    }


def evaluate_case(case, p, scope, device, output):
    start = perf_counter()
    manifest, validation, boundaries = scope
    config = p["config"]
    checkpoint = p["checkpoints"]["last.pt"]
    if (p["completed_epochs"], p["data"]["sha256"], p["label_policy"],
            config["count_threshold"], config["localization_min_score"]) != (
            40, DATA[case["dataset"]][0], LABEL_POLICY, COUNT_CUTOFF, SCORE_FLOOR):
        raise ValueError("training/evaluation metadata differs")
    if read_json(Path(checkpoint["path"]).with_name("run_manifest.json")) != p:
        raise ValueError("local training run manifest differs from archive")
    state = load_checkpoint(Path(checkpoint["path"]), torch.device("cpu"))
    if (state["config"], state["data_sha256"], state["git_commit"], state["history"][-1]["epoch"]) != (
            config, DATA[case["dataset"]][0], p["git_commit"], 40):
        raise ValueError("checkpoint content differs from frozen provenance")
    del state["history"]
    model = make_model(config, initialize=False)
    model.load_state_dict(state["model"])
    del state
    model.to(device).eval()
    stages = ("early", "final") if case["staged"] else ("final",)
    rows, ranked = ({stage: [] for stage in stages} for _ in range(2))
    torch.cuda.reset_peak_memory_stats(device)
    for record in validation:
        with load_rgb(record, manifest.parent) as image:
            predictions = (predict_image_stages(model, image, config, device) if case["staged"] else
                           {"final": predict_image(model, image, config, device)})
        for stage in stages:
            row, pairs = analyze_image(record, predictions[stage], boundaries)
            rows[stage].append(row)
            ranked[stage].append(pairs)
    torch.cuda.synchronize(device)
    result = {"status": "completed", "full_internal_val_passes": 1, "images": len(validation),
              "seconds": perf_counter() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "bn_buffers": describe_bn(backbone_bn_buffers(model)), "stages": {}}
    for stage in stages:
        path = output / case["dataset"] / str(case["seed"]) / f"{case['arm']}_{stage}.csv"
        write_csv(path, rows[stage])
        result["stages"][stage] = aggregate_stages(rows[stage], ranked[stage]) | {
            "csv": str(path.relative_to(output)), "sha256": file_sha(path),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "loopcrowd-lc002")
    args = parser.parse_args()
    # A fresh output directory prevents a silent second pass after an interrupted run.
    args.output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(["git", "merge-base", "--is-ancestor", BASE, "HEAD"]).strip():
        raise AssertionError("unreachable merge-base output")
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    runs = archive_runs()
    ledger = create_ledger(runs)
    write_csv(args.output / "run_ledger.csv", ledger)
    plan = []
    for dataset, seed, arm, source, staged in planned_cases():
        p = runs[source][0]["provenance"]
        plan.append({"dataset": dataset, "seed": seed, "arm": arm, "historical_run": source, "staged": staged}
                    | checkpoint_columns("e40", p["checkpoints"]["last.pt"]))
    if len(plan) != 33 or len({case["e40_sha256"] for case in plan}) != 33:
        raise ValueError("frozen plan must contain 33 distinct E40 checkpoints")
    write_csv(args.output / "frozen_cases.csv", plan)
    device = torch.device("cuda:0")
    evidence = {"analysis_commit": code, "datasets": {}, "cases": [],
                "device": torch.cuda.get_device_name(device), "torch": str(torch.__version__)}
    scopes = {}
    for dataset in DATA:
        try:
            scopes[dataset] = load_dataset(dataset, args.data_dir)
        except (ValueError, OSError) as error:
            evidence["datasets"][dataset] = {"status": "blocked", "reason": str(error)}
        else:
            manifest, val, boundaries = scopes[dataset]
            evidence["datasets"][dataset] = {
                "status": "verified", "train_density_tertiles": boundaries, "manifest": str(manifest),
                "manifest_sha256": DATA[dataset][0], "ordered_val_ids": [row.image_id for row in val],
            }
    for case in plan:
        print(f"{case['dataset']} {case['seed']} {case['arm']}", flush=True)
        if case["dataset"] not in scopes or case["e40_local"] != "verified":
            result = {"status": "not_run", "reason": "manifest blocked or frozen E40 unavailable",
                      "full_internal_val_passes": 0}
        else:
            run = runs[case["historical_run"]][0]
            result = evaluate_case(case, run["provenance"], scopes[case["dataset"]], device, args.output)
        evidence["cases"].append(case | result)
        write_json(args.output / "evaluation.json", evidence)
        print(result["status"], result.get("seconds"), flush=True)


if __name__ == "__main__":
    main()
