"""Audit and run the six fixed LC-005 paired arms serially on one RTX 3090."""

import argparse
import subprocess
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import default_collate

from loopcrowd.calibration import json_digest
from loopcrowd.data import TrainingCrops
from loopcrowd.experiment import file_sha, make_model, provenance, read_config, seed_all, state_sha, write_json
from loopcrowd.inference import autocast
from loopcrowd.manifest import read_manifest
from loopcrowd.model import pretrained_path
from loopcrowd.scheduling import UpdateSchedule
from loopcrowd.training import gpu_scaler, optimizer_for, supervised_point_loss, train_step
from scripts.run_lc003 import MANIFEST_SHA, WEIGHT_SHA, load_json, run_logged
from scripts.run_lc004a import verify_workload

CONFIGS = {arm: Path(f"configs/lc005_{arm}.json") for arm in ("final_only", "aux_025")}
PAIR_ORDER = {42: ("final_only", "aux_025"), 123: ("aux_025", "final_only"), 2026: ("final_only", "aux_025")}
PROTOCOL = Path("reports/LC-005/protocol.md")
CALIBRATION_IDS = Path("reports/LC-004B/fit_manifest.json")


def graph_sha(loss):
    nodes = [loss.grad_fn]
    indexes = {nodes[0]: 0}
    edges = []
    for node in nodes:
        row = []
        for child, slot in node.next_functions:
            if child is None:
                row.append(None)
                continue
            if child not in indexes:
                indexes[child] = len(nodes)
                nodes.append(child)
            row.append((indexes[child], slot))
        edges.append([type(node).__name__, row])
    return json_digest(edges)


def audit_arm(config, train, manifest, device):
    seed_all(config["seed"])
    model = make_model(config, initialize=True)
    init_sha = state_sha(model)
    model.to(device).train()
    optimizer = optimizer_for(model, config)
    scaler = gpu_scaler(device, config["precision"])
    updates = (len(train) + config["batch_size"] - 1) // config["batch_size"]
    schedule = UpdateSchedule(optimizer, config, updates * config["epochs"])
    rates = []
    for _ in range(schedule.total_updates):
        rates.append(schedule.apply())
        schedule.global_step += 1
    for group in optimizer.param_groups:
        group["lr"] = rates[0]
    crops = TrainingCrops(train, manifest.parent, config, 1)
    images, labels = default_collate([crops[index] for index in range(config["batch_size"])])
    crop_probe = labels.pop("crop_identity").tolist()
    images = images.to(device)
    target = {key: value.to(device) for key, value in labels.items()}
    seen = {}
    watched = {
        "shared_block": model.blocks[0], "fuse": model.fuse,
        "point_logits": model.point_logits, "point_offsets": model.point_offsets,
    }
    handles = [module.register_forward_pre_hook(
        lambda module, _, key=key: seen.setdefault(key, []).append(id(module))
    ) for key, module in watched.items()]
    torch.cuda.reset_peak_memory_stats(device)
    with autocast(device, config["precision"]):
        loss = supervised_point_loss(model, images, target, config)
    signature = graph_sha(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    for handle in handles:
        handle.remove()
    calls = {key: {"calls": len(ids), "unique_objects": len(set(ids))} for key, ids in seen.items()}
    assert calls.keys() == watched.keys()
    assert all(value == {"calls": 2, "unique_objects": 1} for value in calls.values())
    assert torch.isfinite(loss)
    warm_peak = torch.cuda.max_memory_allocated(device)
    warm_reserved = torch.cuda.max_memory_reserved(device)
    elapsed = []
    for _ in range(3):
        torch.cuda.synchronize(device)
        start = perf_counter()
        value = train_step(model, optimizer, scaler, images, target, config, device, diagnostics={})
        torch.cuda.synchronize(device)
        elapsed.append(1000 * (perf_counter() - start))
        assert torch.isfinite(value)
    return {
        "config": config, "config_sha256": json_digest(config), "initial_state_sha256": init_sha,
        "autograd_graph_sha256": signature,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "calls": calls, "probe_crop_index_left_top": crop_probe,
        "updates_per_epoch": updates, "planned_updates": schedule.total_updates,
        "warmup_updates": schedule.warmup_updates, "planned_lr_trajectory_sha256": json_digest(rates),
        "warmup_peak_allocated_bytes": warm_peak, "warmup_peak_reserved_bytes": warm_reserved,
        "train_step_ms_after_warmup": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if file_sha(args.manifest) != MANIFEST_SHA or file_sha(pretrained_path()) != WEIGHT_SHA:
        parser.error("local manifest or cached ResNet18 weights do not match the frozen digests")
    records = read_manifest(args.manifest)
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    if len(train) != 270 or len(val) != 30 or any(
        record.points is None or not (args.manifest.parent / record.image_path).is_file()
        for record in train + val
    ):
        parser.error("missing or incompatible train/validation data")
    fit_ids = load_json(CALIBRATION_IDS)
    assert sum(map(len, fit_ids.values())) == 60
    base = read_config(Path("configs/lc003_warmup_cosine.json"), "shared_2")
    configs = {arm: read_config(path) for arm, path in CONFIGS.items()}
    for (arm, config), weight in zip(configs.items(), (0.0, 0.25), strict=True):
        if config != base | {"early_loss_weight": weight}:
            parser.error(f"unexpected settings for {arm}")
    subprocess.run(["git", "ls-files", "--error-unmatch", str(PROTOCOL), *map(str, CONFIGS.values())],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "HEAD", "--quiet"], check=True)
    device = torch.device("cuda")
    if torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name(device):
        parser.error("the frozen queue expects one visible RTX 3090")
    if not torch.cuda.is_bf16_supported():
        parser.error("the frozen queue requires BF16")
    args.output.mkdir(parents=True)
    provenance_data = provenance(configs["final_only"], device, args.manifest)
    free, total = torch.cuda.mem_get_info(device)
    audit = {
        "execution": "in_progress", "provenance": provenance_data,
        "protocol_sha256": file_sha(PROTOCOL), "train_images": len(train), "validation_images": len(val),
        "config_files": {arm: {"path": str(path), "sha256": file_sha(path)} for arm, path in CONFIGS.items()},
        "pretrained_weight": {"path": str(pretrained_path()), "sha256": WEIGHT_SHA},
        "label_policy": base["label_policy"], "preceding_fit_manifest_sha256": file_sha(CALIBRATION_IDS),
        "calibration_executed": False, "device_total_bytes": total, "device_free_bytes_before_preflight": free,
        "pair_order": PAIR_ORDER, "arms": {},
    }
    audit_path = args.output / "preflight.json"
    write_json(audit_path, audit)
    for seed, order in PAIR_ORDER.items():
        for arm in order:
            print(f"preflight {seed}/{arm}", flush=True)
            key = f"{seed}/{arm}"
            try:
                audit["arms"][key] = audit_arm(configs[arm] | {"seed": seed}, train, args.manifest, device)
            except torch.cuda.OutOfMemoryError:
                audit["execution"] = "blocked_cuda_oom"
                audit["blocked_arm"] = key
                write_json(audit_path, audit)
                raise
            torch.cuda.empty_cache()
            write_json(audit_path, audit)
        a, b = (audit["arms"][f"{seed}/{arm}"] for arm in configs)
        fields = (
            "initial_state_sha256", "autograd_graph_sha256", "parameters", "calls",
            "probe_crop_index_left_top", "updates_per_epoch", "planned_updates", "warmup_updates",
            "planned_lr_trajectory_sha256",
        )
        if any(a[key] != b[key] for key in fields) or a["parameters"] != 1_596_614 or a["planned_updates"] != 5400:
            raise ValueError(f"seed {seed} failed paired compute or initialization preflight")
    audit["execution"] = "passed"
    write_json(audit_path, audit)
    audit_sha = file_sha(audit_path)
    print(f"preflight passed {audit_sha}", flush=True)

    for seed, order in PAIR_ORDER.items():
        histories = {}
        for arm in order:
            output = args.output / "train" / str(seed) / arm
            run_logged(output, "train", (
                "--config", CONFIGS[arm], "--seed", seed, "--manifest", args.manifest, "--device", "cuda",
            ))
            history = load_json(output / "history.json")
            run = load_json(output / "run_manifest.json")
            preflight = audit["arms"][f"{seed}/{arm}"]
            if (
                run["execution"] != "completed" or run["completed_epochs"] != 40 or run["global_step"] != 5400
                or run["initial_state_sha256"] != preflight["initial_state_sha256"]
                or run["config_sha256"] != preflight["config_sha256"]
                or run["git_commit"] != provenance_data["git_commit"]
                or any(e["amp_skipped_updates"] or e["optimizer_updates"] != 135 for e in history)
                or json_digest([lr for e in history for lr in e["learning_rates"]]) != preflight["planned_lr_trajectory_sha256"]
            ):
                raise ValueError(f"training did not match the preflight commitment: {seed}/{arm}")
            histories[arm] = history
        a, b = (histories[arm] for arm in configs)
        verify_workload(a, b)
        assert all(left["crop_order_sha256"] == right["crop_order_sha256"] for left, right in zip(a, b, strict=True))
    write_json(args.output / "queue_complete.json", {
        "execution": "completed", "runs": 6, "epochs": 240, "updates": 32400, "preflight_sha256": audit_sha,
        "training_code_sha": provenance_data["git_commit"],
    })
    print("LC-005 queue completed", flush=True)


if __name__ == "__main__":
    main()
