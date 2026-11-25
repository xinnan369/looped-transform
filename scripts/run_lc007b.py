"""Execute only the frozen LC-007B Stage 1 queue on local resources."""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from loopcrowd.accuracy_diagnosis import density
from loopcrowd.calibration import json_digest
from loopcrowd.ema import ParameterEMA, preserve_training_context
from loopcrowd.experiment import file_sha, make_model, provenance, read_config, seed_all, state_sha, write_json
from loopcrowd.grouped_validation import grouped_validation
from loopcrowd.manifest import Record, read_manifest
from loopcrowd.model import pretrained_path
from loopcrowd.scheduling import UpdateSchedule
from loopcrowd.targets import point_targets
from loopcrowd.training import fit, gpu_scaler, optimizer_for, train_step
from scripts.run_lc006 import PARAMETERS
from scripts.run_lc007a import read_json

FROZEN = Path("configs/lc007b_ema_screen.json")
PROTOCOL = Path("reports/LC-007B/protocol.md")
COMMON = ("p2.", "p3.", "project.", "fuse.", "point_logits.", "point_offsets.")
DEFAULT_RESERVES = {"preflight": 10, "initialization": 2, "training": 9, "validation_live": 7,
                    "validation_ema": 7, "checkpoint": 1, "train_batch": 1, "validation_image": 1}


class StageBudget:
    """Account for synchronized GPU phases and successful experiment updates."""

    def __init__(self, device, limits, ledger):
        self.device, self.limits, self.ledger = device, limits, ledger
        self.elapsed = 0.0
        self.started = None
        self.updates = 0
        self.peak_reserved_bytes = 0
        self.case = "preflight"
        self.times = defaultdict(list)

    def check(self, next_operation=None):
        used = self.elapsed + (perf_counter() - self.started if self.started is not None else 0)
        reserve = (max(DEFAULT_RESERVES[next_operation],
                       max(self.times[next_operation], default=0.0)) if next_operation is not None else 0)
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, torch.cuda.max_memory_reserved(self.device))
        if self.peak_reserved_bytes > self.limits["peak_reserved_bytes"]:
            raise RuntimeError("Stage 1 process peak reserved memory exceeded the 2 GiB cap")
        if used + reserve >= self.limits["gpu_seconds"]:
            raise RuntimeError(f"Stage 1 GPU time cap: {used:.3f}s used, {reserve:.3f}s for next operation")
        if next_operation == "train_batch" and self.updates >= self.limits["total_updates"]:
            raise RuntimeError("Stage 1 optimizer update budget reached")

    def record_update(self):
        self.updates += 1
        if self.updates > self.limits["total_updates"]:
            raise RuntimeError("Stage 1 optimizer update cap exceeded")

    @contextmanager
    def phase(self, name):
        self.check(name)
        torch.cuda.synchronize(self.device)
        self.started = perf_counter()
        try:
            yield
            torch.cuda.synchronize(self.device)
            self.check()
        finally:
            end_error = None
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as error:
                end_error = str(error)
            seconds = perf_counter() - self.started
            self.started = None
            self.elapsed += seconds
            self.times[name].append(seconds)
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, torch.cuda.max_memory_reserved(self.device))
            with self.ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "case": self.case, "phase": name, "seconds": seconds, "cumulative_gpu_seconds": self.elapsed,
                    "successful_project_updates": self.updates, "peak_reserved_bytes": self.peak_reserved_bytes,
                    "sync_error": end_error,
                }, allow_nan=False) + "\n")

    def totals(self):
        return {"gpu_seconds": self.elapsed, "peak_reserved_bytes": self.peak_reserved_bytes,
                "successful_project_updates": self.updates,
                "seconds_by_phase": {key: sum(values) for key, values in self.times.items()}}


def frozen_inputs(path):
    frozen = read_json(FROZEN)
    if file_sha(path) != frozen["manifest_sha256"] or file_sha(pretrained_path()) != frozen["weight_sha256"]:
        raise ValueError("local B manifest or ResNet18 checksum differs")
    data = read_manifest(path, splits=("train", "val"))
    train = [r for r in data if r.split == "train"]
    val = [r for r in data if r.split == "val"]
    sizes = {key: [len(rows), sum(len(row.points) for row in rows)] for key, rows in (("train", train), ("val", val))}
    if sizes != {"train": [360, 44856], "val": [40, 4295]} or len({row.dataset for row in data}) != 1:
        raise ValueError("B split/image/cleaned head sizes differ")
    boundaries = np.quantile([density(r) for r in train], (1 / 3, 2 / 3)).tolist()
    np.testing.assert_allclose(boundaries, (82.65177408854167, 180.98619249131943), atol=1e-12, rtol=0)
    evidence = {"images_and_cleaned_heads": sizes, "train_only_density_tertiles": boundaries,
                "ordered_val_ids": [row.image_id for row in val], "manifest_path": str(path),
                "manifest_sha256": frozen["manifest_sha256"], "pretrained_weight_path": str(pretrained_path()),
                "pretrained_weight_sha256": frozen["weight_sha256"]}
    return frozen, evidence


def state_digest(value):
    """Hash the nested disposable optimizer state or a sequence of dense gradient tensors."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key, val in item.items():
                visit(key)
                visit(val)
        elif isinstance(item, (list, tuple)):
            for val in item:
                visit(val)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def disposable_probe(config, initial_samples, validation, device, guard, *, enabled):
    seed_all(config["seed"])
    model = make_model(config, initialize=True)
    starts = {"initial_sha256": state_sha(model), "common_sha256": state_sha(model, prefixes=COMMON),
              "block_sha256": [state_sha(block) for block in model.blocks],
              "parameters": sum(p.numel() for p in model.parameters())}
    model.to(device).train()
    optimizer = optimizer_for(model, config)
    scaler = gpu_scaler(device, config["precision"])
    schedule = UpdateSchedule(optimizer, config, 7200)
    ema = ParameterEMA(model, config["ema_decay"]) if enabled else None
    images, labels = initial_samples
    signature, step_seconds = [], []
    for _ in range(3):
        guard.check("train_batch")
        rate = schedule.apply()
        details = {}
        torch.cuda.synchronize(device)
        start = perf_counter()
        loss = train_step(model, optimizer, scaler, images, labels, config, device, details, require_finite=True)
        schedule.global_step += 1
        if ema is not None:
            ema.update(updated=True)
            ema.require_finite(check_live=True)
        torch.cuda.synchronize(device)
        step_seconds.append(perf_counter() - start)
        signature.append({
            "rate": rate, "loss": float(loss), "diagnostics": {k: float(v) for k, v in details.items()},
            "gradients_sha256": state_digest([p.grad for p in model.parameters()]),
            "live_parameters_and_buffers_sha256": state_sha(model),
            "optimizer_sha256": state_digest(optimizer.state_dict()),
            "rng_sha256": state_digest((torch.get_rng_state(), torch.cuda.get_rng_state(device))),
        })
        guard.check()
    if ema is not None:
        row, directory = validation
        before = state_sha(model)
        before_rng = state_digest((torch.get_rng_state(), torch.cuda.get_rng_state(device)))
        with preserve_training_context(model, device), ema.use_shadow():
            grouped_validation(model, [row], directory, config, device, staged=config["early_loss_weight"] is not None,
                               boundaries=(82.65177408854167, 180.98619249131943),
                               check=lambda: guard.check("validation_image"))
        if state_sha(model) != before or not model.training or before_rng != state_digest(
                (torch.get_rng_state(), torch.cuda.get_rng_state(device))):
            raise ValueError("preflight EMA validation changed live parameters, buffers, mode or RNG")
        if ema.step != schedule.global_step:
            raise ValueError("synthetic EMA position differs from optimizer schedule")
    schedule.global_step = 0
    rates = []
    for _ in range(schedule.total_updates):
        rates.append(schedule.apply())
        schedule.global_step += 1
    return starts | {"trace": signature, "step_seconds": step_seconds,
                     "lr_sha256": json_digest(rates), "warmup_updates": schedule.warmup_updates}


def preflight(provenance_data, configs, keys, output, device, guard):
    result = dict(provenance_data)
    result.update(execution="in_progress", cases={}, disposable_successful_updates=0)
    path = output / "preflight.json"
    write_json(path, result)
    fixture = output / "synthetic_validation.png"
    Image.new("RGB", (1024, 768), (141, 21, 76)).save(fixture)
    validation = Record("lc007b_synthetic", "synthetic", "val", fixture.name, 1024, 768, [[30., 46.], [51., 62.]])
    for seed, arm in keys:
        config = configs[seed, arm]
        key = f"{seed}/{arm}"
        guard.case = f"preflight/{key}"
        print(f"preflight {key}", flush=True)
        with guard.phase("preflight"):
            g = torch.Generator().manual_seed(19)
            images = torch.rand((2, 3, 512, 512), generator=g).to(device)
            sample = point_targets([[32., 44.], [33., 45.], [261., 182.]], 512, 512)
            labels = {name: value.unsqueeze(0).repeat((2,) + (1,) * value.ndim).to(device)
                      for name, value in sample.items()}
            off, on = [disposable_probe(config, (images, labels), (validation, output), device, guard, enabled=mode)
                       for mode in (False, True)]
            del images, labels
        comparison = {key for key in off if key != "step_seconds"}
        if any(off[key] != on[key] for key in comparison):
            diff = sorted(key for key in comparison if off[key] != on[key])
            raise ValueError(f"EMA on/off synthetic training trace differs for {seed}/{arm}: {diff}")
        if on["parameters"] != PARAMETERS[config["variant"]] or on["warmup_updates"] != 360:
            raise ValueError("synthetic preflight parameter or update budget mismatch")
        result["cases"][key] = {**on, "off_step_seconds": off["step_seconds"], "exact_on_off_trace": True,
                                "config_sha256": json_digest(config)}
        result["disposable_successful_updates"] += 6
        write_json(path, result)
        torch.cuda.empty_cache()
    for seed in (42, 123):
        paired = [result["cases"][f"{s}/{arm}"] for s, arm in keys if s == seed]
        if (len({row["common_sha256"] for row in paired}) != 1
                or len({row["block_sha256"][0] for row in paired if row["block_sha256"]}) != 1
                or len({row["lr_sha256"] for row in paired}) != 1
                or any(len(row["block_sha256"]) != len(set(row["block_sha256"])) for row in paired)):
            raise ValueError(f"paired preflight initialization or LR differs in seed {seed}")
    result["execution"] = "passed"
    write_json(path, result)
    return result


def verified_run(run, history, initial, frozen, revision):
    if (run["execution"] != "completed" or run["completed_epochs"] != 40 or run["global_step"] != 7200
            or [row["epoch"] for row in history] != list(range(1, 41))
            or run["data"]["sha256"] != frozen["manifest_sha256"] or run["tracked_changes"]
            or run["git_commit"] != revision or run["initial_state_sha256"] != initial["initial_sha256"]
            or run["config_sha256"] != initial["config_sha256"]
            or run["weight_initialization"]["sha256"] != frozen["weight_sha256"]
            or any(row["optimizer_updates"] != 180 or row["amp_skipped_updates"] or
                   row["ema_step"] != 180 * row["epoch"] for row in history)
            or json_digest([lr for row in history for lr in row["learning_rates"]]) != initial["lr_sha256"]):
        raise ValueError("training evidence differs from the frozen Stage 1 budget or provenance")
    return [row["crop_order_sha256"] for row in history]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path.home() / "loopcrowd-lc002/shanghaitech_b.jsonl")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    progress = {"execution": "in_progress", "ordered_runs": []}
    progress_path = args.output / "queue_progress.json"
    write_json(progress_path, progress)
    guard = None
    try:
        frozen, data = frozen_inputs(args.manifest)
        paths = ["git", "ls-files", "--error-unmatch", str(FROZEN), str(PROTOCOL), *frozen["configs"].values()]
        subprocess.run(paths, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "merge-base", "--is-ancestor", frozen["parent"], "HEAD"], check=True)
        subprocess.run(["git", "diff", "HEAD", "--quiet"], check=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        progress["training_code_sha"] = revision
        keys = [(int(seed), arm) for seed, order in frozen["order"].items() for arm in order]
        configs = {(seed, arm): read_config(Path(frozen["configs"][arm])) |
                   {"seed": seed, "ema_decay": frozen["ema_decay"]} for seed, arm in keys}
        for (seed, arm), config in configs.items():
            expected = read_config(Path("configs/lc006_shared_2_aux_025.json")) | {
                "variant": arm.removesuffix("_aux_025"), "seed": seed, "ema_decay": .99,
                "early_loss_weight": .25 if arm.endswith("_aux_025") else None,
            }
            if config != expected:
                raise ValueError(f"unexpected matching recipe for {seed}/{arm}")
        device = torch.device("cuda:0")
        if (torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name(device)
                or not torch.cuda.is_bf16_supported()):
            raise ValueError("Stage 1 requires exactly one visible BF16-capable RTX 3090")
        guard = StageBudget(device, frozen["limits"], args.output / "gpu_phases.jsonl")
        torch.cuda.reset_peak_memory_stats(device)
        preflight_input = data | {
            "provenance": provenance(configs[keys[0]], device, args.manifest), "frozen": frozen,
            "protocol_sha256": file_sha(PROTOCOL), "config_index_sha256": file_sha(FROZEN),
            "arm_configs": {arm: {"path": path, "sha256": file_sha(Path(path))}
                            for arm, path in frozen["configs"].items()},
            "gpu_memory_start": list(torch.cuda.mem_get_info(device)),
        }
        audit = preflight(preflight_input, configs, keys, args.output, device, guard)
        progress["preflight_sha256"] = file_sha(args.output / "preflight.json")
        print("Preflight passed", flush=True)
        crops_by_seed = {}
        for seed, arm in keys:
            guard.case = f"{seed}/{arm}"
            progress["current_run"] = guard.case
            write_json(progress_path, progress | guard.totals())
            if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != revision:
                raise ValueError("working branch HEAD changed after protocol freeze")
            print(f"train {guard.case}", flush=True)
            output = args.output / "train" / str(seed) / arm
            run = fit(configs[seed, arm], args.manifest, output, device, guard=guard)
            history = read_json(output / "history.json")
            crops = verified_run(run, history, audit["cases"][guard.case], frozen, revision)
            if crops_by_seed.setdefault(seed, crops) != crops:
                raise ValueError(f"actual paired crop sequence differs for {guard.case}")
            progress["ordered_runs"].append({"seed": seed, "arm": arm, "epochs": 40, "updates": 7200})
            print(f"completed {guard.case} at {guard.elapsed:.2f}s and {guard.updates} updates", flush=True)
            write_json(progress_path, progress | guard.totals())
            torch.cuda.empty_cache()
        if len(progress["ordered_runs"]) != frozen["limits"]["runs"] or guard.updates != frozen["limits"]["total_updates"]:
            raise ValueError("Stage 1 completed with an incorrect experiment denominator")
        progress.update(execution="completed", current_run=None)
    except Exception as error:
        progress.update(execution="stopped", reason=f"{type(error).__name__}: {error}")
        preflight_path = args.output / "preflight.json"
        if preflight_path.is_file():
            preflight_record = read_json(preflight_path)
            if preflight_record["execution"] == "in_progress":
                preflight_record.update(execution="stopped", reason=progress["reason"])
                write_json(preflight_path, preflight_record)
        raise
    finally:
        write_json(progress_path, progress | (guard.totals() if guard is not None else {}))


if __name__ == "__main__":
    main()
