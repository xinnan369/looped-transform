"""Audit and run the preregistered LC-006 ShanghaiTech B matrix serially."""

import argparse
import subprocess
from pathlib import Path

import torch

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, provenance, read_config, write_json
from loopcrowd.manifest import read_manifest
from scripts.audit_lc006_data import MANIFEST_B_SHA, audit_data
from scripts.run_lc003 import WEIGHT_SHA, load_json, run_logged
from scripts.run_lc004a import verify_workload
from scripts.run_lc005 import audit_arm

ARMS = {
    "cnn": ("cnn_only", None), "shared_1": ("shared_1", None),
    "shared_2_final_only": ("shared_2", 0.0), "shared_2_aux_025": ("shared_2", 0.25),
    "unshared_2_final_only": ("unshared_2", 0.0), "unshared_2_aux_025": ("unshared_2", 0.25),
}
CONFIGS = {arm: Path(f"configs/lc006_{arm}.json") for arm in ARMS}
PAIR_ORDER = {
    42: tuple(ARMS),
    123: ("shared_1", "cnn", "unshared_2_aux_025", "unshared_2_final_only", "shared_2_aux_025", "shared_2_final_only"),
    2026: tuple(ARMS),
}
PROTOCOL = Path("reports/LC-006/protocol.md")
DATA_AUDIT = Path("reports/LC-006/data_audit.json")
PROFILE_ARMS = ("cnn", "shared_1", "shared_2_aux_025", "unshared_2_aux_025")
PARAMETERS = {"cnn_only": 1_151_750, "shared_1": 1_596_614, "shared_2": 1_596_614, "unshared_2": 2_041_478}
GPU_QUERY = ("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm", "--format=csv,noheader")


def check_preflight_pair(seed, arms):
    records = [arms[f"{seed}/{arm}"] for arm in ARMS]
    common = ("common_initializer_sha256", "planned_lr_trajectory_sha256", "probe_crop_index_left_top",
              "updates_per_epoch", "planned_updates", "warmup_updates")
    if any(row[key] != records[0][key] for row in records for key in common):
        raise ValueError(f"seed {seed} differs in its common initialization, crop probe or LR horizon")
    recurrent = [row for row in records if row["block_initializer_sha256"]]
    if len({row["block_initializer_sha256"][0] for row in recurrent}) != 1:
        raise ValueError(f"seed {seed} has different first-block initial weights across architectures")
    for variant in ("shared_2", "unshared_2"):
        off, on = (arms[f"{seed}/{variant}_{arm}"] for arm in ("final_only", "aux_025"))
        fields = ("initial_state_sha256", "autograd_graph_sha256", "block_initializer_sha256", "calls")
        if any(off[key] != on[key] for key in fields):
            raise ValueError(f"seed {seed}/{variant} differs in full initialization or compute graph")
        expected = 1 if variant == "shared_2" else 2
        if len(set(on["block_initializer_sha256"])) != expected:
            raise ValueError(f"{variant} has unexpected block initialization identity")


def check_training(run, history, preflight, revision, arm):
    if (
        run["execution"] != "completed" or run["completed_epochs"] != 40 or run["global_step"] != 7200
        or len(history) != 40 or run["data"]["sha256"] != MANIFEST_B_SHA or run["tracked_changes"]
        or run["initial_state_sha256"] != preflight["initial_state_sha256"]
        or run["config_sha256"] != preflight["config_sha256"] or run["git_commit"] != revision
        or run["weight_initialization"]["sha256"] != WEIGHT_SHA
        or any(e["amp_skipped_updates"] or e["optimizer_updates"] != 180 for e in history)
        or json_digest([lr for e in history for lr in e["learning_rates"]]) != preflight["planned_lr_trajectory_sha256"]
    ):
        raise ValueError(f"training did not match the 7,200-update preflight contract: {run['seed']}/{arm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = audit_data(args.manifest, args.root, args.rejections)
    if data != load_json(DATA_AUDIT):
        raise ValueError("fresh B file and capacity audit differs from the frozen preflight evidence")
    train = read_manifest(args.manifest, splits=("train",))
    base = read_config(Path("configs/lc005_final_only.json"))
    configs = {}
    for arm, (variant, weight) in ARMS.items():
        config = read_config(CONFIGS[arm])
        if config != base | {"variant": variant, "early_loss_weight": weight, "audit_training": True}:
            raise ValueError(f"unexpected configuration for {arm}")
        configs[arm] = config
    subprocess.run(["git", "ls-files", "--error-unmatch", str(PROTOCOL), str(DATA_AUDIT), *map(str, CONFIGS.values())],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "HEAD", "--quiet"], check=True)
    device = torch.device("cuda", 0)
    if torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name(device):
        parser.error("LC-006 expects one visible RTX 3090")
    if not torch.cuda.is_bf16_supported():
        parser.error("the GPU must support the frozen BF16 protocol")
    args.output.mkdir(parents=True)
    write_json(args.output / "data_audit.json", data)
    free, total = torch.cuda.mem_get_info(device)
    preflight = {
        "execution": "in_progress", "protocol_sha256": file_sha(PROTOCOL),
        "provenance": provenance(configs["cnn"], device, args.manifest), "pair_order": PAIR_ORDER,
        "data_audit_sha256": file_sha(args.output / "data_audit.json"), "manifest": data["manifest"],
        "pretrained_weight": data["pretrained_weight"],
        "config_files": {arm: {"path": str(path), "sha256": file_sha(path)} for arm, path in CONFIGS.items()},
        "gpu_total_bytes": total, "gpu_free_bytes_before_preflight": free, "arms": {}, "profiles": {},
    }
    preflight_path = args.output / "preflight.json"
    write_json(preflight_path, preflight)
    for seed, order in PAIR_ORDER.items():
        for arm in order:
            key = f"{seed}/{arm}"
            print(f"preflight {key}", flush=True)
            try:
                result = audit_arm(configs[arm] | {"seed": seed}, train, args.manifest, device)
            except torch.cuda.OutOfMemoryError:
                preflight["execution"] = "blocked_cuda_oom"
                preflight["blocked_arm"] = key
                write_json(preflight_path, preflight)
                raise
            if ((result["updates_per_epoch"], result["planned_updates"], result["warmup_updates"]) != (180, 7200, 360)
                    or result["parameters"] != PARAMETERS[ARMS[arm][0]]):
                raise ValueError(f"unexpected preflight budget or parameter count: {key}")
            preflight["arms"][key] = result
            torch.cuda.empty_cache()
            write_json(preflight_path, preflight)
        check_preflight_pair(seed, preflight["arms"])
    for arm in PROFILE_ARMS:
        variant = ARMS[arm][0]
        output = args.output / "profile" / variant
        gpu_state = subprocess.check_output(GPU_QUERY, text=True).strip()
        run_logged(output, "profile", (
            "--config", CONFIGS[arm], "--device", "cuda", "--manifest", args.manifest, "--repeats", 9,
        ))
        preflight["profiles"][variant] = {
            "profile_config": arm, "gpu_before_utilization_memory_temperature_clock": gpu_state,
            "profile": load_json(output / "profile.json"), "run": load_json(output / "run_manifest.json"),
            "local_log": {"path": str(output.with_suffix(".log")), "sha256": file_sha(output.with_suffix(".log"))},
        }
    preflight["execution"] = "passed"
    write_json(preflight_path, preflight)
    revision = preflight["provenance"]["git_commit"]
    progress = {"execution": "in_progress", "ordered_runs": [], "preflight_sha256": file_sha(preflight_path),
                "training_code_sha": revision}
    progress_path = args.output / "queue_progress.json"
    write_json(progress_path, progress)
    print(f"preflight passed {progress['preflight_sha256']}", flush=True)
    for seed, order in PAIR_ORDER.items():
        reference = None
        for arm in order:
            output = args.output / "train" / str(seed) / arm
            run_logged(output, "train", (
                "--config", CONFIGS[arm], "--seed", seed, "--manifest", args.manifest, "--device", "cuda",
            ))
            run, history = (load_json(output / f"{name}.json") for name in ("run_manifest", "history"))
            check_training(run, history, preflight["arms"][f"{seed}/{arm}"], revision, arm)
            if reference is None:
                reference = history
            else:
                verify_workload(reference, history)
                if [row["crop_order_sha256"] for row in reference] != [row["crop_order_sha256"] for row in history]:
                    raise ValueError(f"actual crop identities differ from the seed reference: {seed}/{arm}")
            progress["ordered_runs"].append({"seed": seed, "arm": arm, "epochs": 40, "updates": 7200})
            write_json(progress_path, progress)
    progress["execution"] = "completed"
    progress["epochs"] = sum(row["epochs"] for row in progress["ordered_runs"])
    progress["updates"] = sum(row["updates"] for row in progress["ordered_runs"])
    write_json(progress_path, progress)
    print("LC-006 queue completed", flush=True)


if __name__ == "__main__":
    main()
