"""Freeze or execute the separately authorized LC-007H guided training queue."""

import argparse
import shutil
import signal
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, write_json
from loopcrowd.grouped_validation import GainStatistics
from loopcrowd.training import fit
from scripts.archive_lc007c import check_training
from scripts.diagnose_lc007h import G, archives
from scripts.prepare_lc007e import config_for, trajectory_columns
from scripts.prepare_lc007h import SCREEN as PROPOSAL, audit
from scripts.run_lc007d import ScreenBudget, background_snapshot, git, read_json, require_authorization
from scripts.run_lc007h_e40 import stop_at_deadline

SCREEN = Path("configs/lc007h_training.json")


class TrainingBudget(ScreenBudget):
    label = "LC-007H training"
    cleanup_reserve_seconds = 25

    def __init__(self, limits, ledger, device):
        super().__init__(limits, ledger, device)
        if "peak_reserved_bytes" in limits:
            raise ValueError("LC-007H observes actual GPU use without a quota")
        self.longest.update(cpu_preflight=60, device_preflight=10, initialization=5,
                            training=20, validation_live=20, validation_ema=20, checkpoint=5)


def require_training_authorization(owner, screen, digest, revision, tree):
    require_authorization(owner, screen, digest, revision)
    if owner["approval_scope"] != screen["scope"] or owner.get("source_tree") != tree:
        raise ValueError("authorization does not cover this training scope and source tree")


def training_config(screen, seed, arm):
    return config_for(screen, seed, arm) | {"refinement": screen["arms"][arm]["refinement"]}


def gain_cpu_proxy():
    """Synthetic 240-tile full-size sweep, including global and density accumulators."""
    values = np.linspace(.90, 1.10, 240 * 64 * 64, dtype=np.float32).reshape(240, 64, 64)
    elapsed = []
    for _ in range(5):
        all_tiles, density = GainStatistics(), GainStatistics()
        start = perf_counter()
        for tile in values:
            all_tiles.add(tile, 512, 512)
            density.add(tile, 512, 512)
        all_tiles.result()
        density.result()
        elapsed.append(perf_counter() - start)
    return elapsed


def verify_inputs(screen):
    for path, digest in screen["inputs_sha256"].items():
        if file_sha(path) != digest:
            raise ValueError(f"frozen training input differs: {path}")
    proposal = read_json(PROPOSAL)
    if (screen["order"] != proposal["order"]
            or any(screen["arms"][arm]["refinement"] != spec["refinement"]
                   or screen["arms"][arm]["comparison"] != spec["reference"] for arm, spec in proposal["arms"].items())):
        raise ValueError("training matrix differs from the accepted proposal")
    return proposal


def preflight(manifest, screen):
    proposal = verify_inputs(screen)
    inspected = audit(manifest, proposal)
    frozen = read_json(Path("reports/LC-007H/initialization_audit.json"))
    for key in ("historical_execution_commit", "planned_initializations", "proposal_sha256"):
        if inspected[key] != frozen[key]:
            raise ValueError(f"training preflight diverges at {key}")
    # Paths may be mounted differently. Content identity and ordered data must agree.
    for key in ("manifest_sha256", "pretrained_weight_sha256", "ordered_val_ids",
                "train_only_density_tertiles", "images_and_cleaned_heads"):
        if inspected["data"][key] != frozen["data"][key]:
            raise ValueError(f"training split or weight diverges at {key}")
    tables = archives()
    for seed_text, arms in screen["order"].items():
        seed = int(seed_text)
        for arm in arms:
            start = inspected["planned_initializations"][f"{seed}/{arm}"]
            config = training_config(screen, seed, arm)
            if (json_digest(config) != start["config_sha256"] or config["epochs"] !=
                    screen["limits_if_approved"]["epochs_per_run"] or config.get("depth_readout_start_epoch", 1) != 1):
                raise ValueError(f"effective configuration differs at {seed}/{arm}")
            check_training(tables[seed, start["reference"], "training"])
    samples = gain_cpu_proxy()
    limits = screen["limits_if_approved"]
    historical = inspected["resource_proxy"]
    # The prior projection already includes existing validation and a 25% reserve.
    # Add the measured worst synthetic CPU sweep cost again for all new validation sweeps.
    cpu_extra = max(samples) * 2 * limits["total_epochs"]
    projected = historical["conditional_training_with_25pct_allowance_and_120s_seconds"] + cpu_extra
    resource = {"historical": historical, "gain_cpu_seconds_per_sweep_samples": samples,
                "gain_cpu_allowance_seconds": cpu_extra, "plan_seconds": projected,
                "required_device_headroom_bytes_estimate": 2 * read_json(G / "run_evidence.json")["execution"]["peak_reserved_bytes"],
                "actual_new_gpu": "not_run"}
    evidence = {"plan_sha256": file_sha(SCREEN), "arm_config_sha256": file_sha(screen["arm_config"]),
                "planned_starts": inspected["planned_initializations"], "data": inspected["data"],
                "historical_execution_commit": inspected["historical_execution_commit"], "resource_projection": resource}
    return evidence, tables


def execute(manifest, output, screen, revision, tree):
    output.mkdir()
    limits = screen["limits_if_approved"]
    device = torch.device("cuda:0")
    budget = TrainingBudget(limits, output / "gpu_phases.jsonl", device)
    progress = {"execution": "in_progress", "code_commit": revision, "source_tree": tree, "completed": [],
                "config_sha256": file_sha(SCREEN), "external_download_bytes": 0, "current": None}
    try:
        with budget.phase("cpu_preflight"):
            if shutil.disk_usage(output).free < screen["disk_free_bytes"]:
                raise RuntimeError("insufficient local disk for training evidence")
            evidence, tables = preflight(manifest, screen)
            write_json(output / "preflight_cpu.json", evidence)
            if evidence["resource_projection"]["plan_seconds"] > limits["gpu_seconds"]:
                raise RuntimeError("frozen projection exceeds 7200 seconds and requires a revised proposal")
        with budget.phase("device_preflight"):
            (output / "background_start.txt").write_text(background_snapshot(), encoding="utf-8")
            if torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name(device) or not torch.cuda.is_bf16_supported():
                raise RuntimeError("the frozen BF16 RTX 3090 environment is unavailable")
            budget.active = True
            free, total = torch.cuda.mem_get_info(device)
            write_json(output / "device_memory_start.json", {"free_bytes": free, "total_bytes": total})
            if free < evidence["resource_projection"]["required_device_headroom_bytes_estimate"]:
                raise RuntimeError("insufficient observed physical device headroom")
        for seed_text, arms in screen["order"].items():
            for arm in arms:
                if git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "--untracked-files=no"):
                    raise RuntimeError("the frozen worktree changed")
                seed, label = int(seed_text), f"{seed_text}/{arm}"
                budget.case, progress["current"] = label, label
                write_json(output / "queue_progress.json", progress | budget.totals())
                start = evidence["planned_starts"][label]
                expected = {"initial_sha256": start["initial_sha256"],
                            "trajectory": trajectory_columns(tables[seed, start["reference"], "training"])}
                run = fit(training_config(screen, seed, arm), manifest, output / "train" / label, device,
                          guard=budget, expected=expected)
                if run["completed_epochs"] != limits["epochs_per_run"] or run["global_step"] != (
                        limits["epochs_per_run"] * limits["updates_per_epoch"]):
                    raise RuntimeError("case completed with a different epoch or successful update count")
                progress["completed"].append(label)
                write_json(output / "queue_progress.json", progress | budget.totals())
        with budget.phase("cleanup"):
            (output / "background_end.txt").write_text(background_snapshot(), encoding="utf-8")
            free, total = torch.cuda.mem_get_info(device)
            write_json(output / "device_memory_end.json", {"free_bytes": free, "total_bytes": total})
        if budget.updates != limits["total_updates"] or budget.totals()["charged_seconds"] > limits["gpu_seconds"]:
            raise RuntimeError("resource ledger differs from the frozen totals")
        progress.update(execution="completed", current=None)
    except BaseException as error:
        progress.update(execution="stopped", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "queue_progress.json", progress | budget.totals())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--authorization", type=Path)
    mode.add_argument("--cpu-verify-only", action="store_true")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    screen = read_json(SCREEN)
    if args.cpu_verify_only:
        evidence, _ = preflight(args.manifest, screen)
        write_json(args.output, evidence)
        return
    revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    require_training_authorization(read_json(args.authorization), screen, file_sha(SCREEN), revision, tree)
    git("merge-base", "--is-ancestor", screen["parent"], revision)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("a clean frozen checkout is required")
    signal.signal(signal.SIGTERM, stop_at_deadline)
    execute(args.manifest, args.output, screen, revision, tree)


if __name__ == "__main__":
    main()
