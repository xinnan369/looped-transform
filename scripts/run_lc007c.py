"""Run the explicitly approved, read-only LC-007C GPU endpoint probes."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import default_collate

from loopcrowd.calibration import json_digest
from loopcrowd.data import TrainingCrops
from loopcrowd.experiment import file_sha, load_checkpoint, make_model, read_config, state_sha, write_json
from loopcrowd.gradient_diagnosis import TOLERANCES, WEIGHT, ZERO_NORM, _diagnose_two_calls, all_finite
from loopcrowd.manifest import read_manifest
from loopcrowd.model import LoopCrowd
from loopcrowd.probe_budget import ProbeBudget
from loopcrowd.targets import point_targets
from scripts.prepare_lc007c import CROP_EPOCH, MANIFEST_SHA, PILOT_CAP, WEIGHT_SHA, crop_plan, verify_checkpoints
from scripts.run_lc007a import read_json
from scripts.run_lc007b import state_digest

DRAFT = "297401f0eba4f8ef90185b8c4db3b478be488a3e"
PROBE = Path("reports/LC-007C/probe_manifest.json")
PROBE_SHA = "aa9ca77267d849d60c9c767139306bdca8ed43aeb5015756f168ff215dc37957"
APPROVAL = "LC-007C-96-1200-2GiB-zero-updates-zero-downloads"
POLICIES = ("live", "ema")
RELEASE_RESERVE = 12


def verified_inputs():
    if file_sha(PROBE) != PROBE_SHA:
        raise ValueError("approved probe manifest SHA-256 differs")
    plan = read_json(PROBE)
    if (plan["tolerances_abs_rel"] != json.loads(json.dumps(TOLERANCES))
            or plan["aux_loss_weight"] != WEIGHT or plan["fp32_near_zero_norm"] != ZERO_NORM
            or plan["proposed_limits"] != PILOT_CAP or plan["planned_batch_policy_cases"] != 96
            or tuple(plan["policies"]) != POLICIES):
        raise ValueError("frozen precision, tolerances, or resources differ")
    manifest = Path(plan["dataset"]["path"])
    weight = Path(plan["weight_cache"]["path"])
    if file_sha(manifest) != MANIFEST_SHA or file_sha(weight) != WEIGHT_SHA:
        raise ValueError("approved dataset or initialization cache SHA-256 differs")
    records = read_manifest(manifest, splits=("train",))
    if len(records) != 360 or sum(len(row.points) for row in records) != 44_856:
        raise ValueError("frozen train population differs")
    if plan["plans"] != [crop_plan(records, seed) for seed in (42, 123)] or plan["checkpoints"] != verify_checkpoints():
        raise ValueError("approved checkpoint references or crop layout differ")
    return plan, manifest, records, weight


def checkpoint_state(ref, plan):
    path = Path(ref["checkpoint"]["path"])
    if file_sha(path) != ref["checkpoint"]["sha256"]:
        raise ValueError("checkpoint changed since frozen verification")
    config_path = Path(ref["source_config"])
    config = read_config(config_path) | {"seed": ref["seed"], "ema_decay": .99}
    if file_sha(config_path) != ref["source_config_sha256"] or json_digest(config) != ref["effective_config_sha256"]:
        raise ValueError("frozen run configuration differs")
    state = load_checkpoint(path, torch.device("cpu"))
    steps = (state["global_step"], state["scheduler"]["global_step"], state["ema"]["step"])
    if (state["config"] != config or state["git_commit"] != plan["checkpoint_training_commit"]
            or state["data_sha256"] != MANIFEST_SHA or state["weight_initialization"]["sha256"] != WEIGHT_SHA
            or state["history"][-1]["epoch"] != 40 or steps != (7200, 7200, 7200) or state["ema"]["decay"] != .99):
        raise ValueError("E40 checkpoint contents differ from frozen provenance")
    model = make_model(config, initialize=False)
    model.load_state_dict(state["model"], strict=True)
    shadow = state["ema"]["shadow"]
    named = dict(model.named_parameters())
    if shadow.keys() != named.keys() or any(
        value.dtype != torch.float32 or value.shape != named[key].shape for key, value in shadow.items()
    ):
        raise ValueError("EMA checkpoint schema differs")
    if not all_finite(t for t in (*state["model"].values(), *shadow.values()) if t.is_floating_point()):
        raise FloatingPointError("nonfinite checkpoint weights or BN buffers")
    return path, config, state, model, shadow


def rng_sha(device):
    python_state = random.getstate()
    algorithm, samples, position, has_gaussian, gaussian = np.random.get_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return state_digest((python_state, algorithm, samples.tobytes(), position, has_gaussian, gaussian,
                         torch.get_rng_state(), cuda))


def model_identity(model):
    tensors = (*model.parameters(), *model.buffers())
    return (tuple((id(tensor), tensor._version) for tensor in tensors),
            tuple(module.training for module in model.modules()),
            tuple(parameter.grad is None for parameter in model.parameters()))


def immutable_readout(model, shadow, images, target, policy, device):
    before = model_identity(model), tuple(value._version for value in shadow.values()), rng_sha(device)
    if not all(before[0][2]):
        raise ValueError("read-only probe received an existing parameter gradient")
    try:
        return _diagnose_two_calls(model, images.to(device), {key: value.to(device) for key, value in target.items()},
                                   precision="bf16", parameter_values=shadow if policy == "ema" else None)
    finally:
        after = model_identity(model), tuple(value._version for value in shadow.values()), rng_sha(device)
        if after != before:
            raise RuntimeError("probe changed live model references, buffers, modes, EMA or RNG")


def sample_batch(crops, positions, batch):
    samples = []
    for crop in batch["crops"]:
        index = positions[crop["image_id"]]
        image, target = crops[index]
        actual = target.pop("crop_identity").tolist()
        valid = int(target["valid"].sum())
        if (actual != [index, crop["left"], crop["top"]] or int(target["heads"]) != crop["crop_gt"]
                or valid != ((crop["valid_width"] + 3) // 4) * ((crop["valid_height"] + 3) // 4)):
            raise ValueError("loaded train crop differs from the frozen origin or target")
        samples.append((image, target))
    images, targets = default_collate(samples)
    return images, targets


@contextmanager
def cuda_checkpoint(ref, plan, device, guard, ledger):
    label = f"{ref['seed']}/{ref['arm']}"
    path = Path(ref["checkpoint"]["path"])
    model = shadow = gpu_shadow = state = None
    try:
        with guard.phase(f"load/{label}", estimate_seconds=15 + RELEASE_RESERVE):
            path, config, state, model, shadow = checkpoint_state(ref, plan)
            initial = {"model_sha256": state_sha(model), "ema_sha256": state_digest(state["ema"]),
                       "optimizer_sha256": state_digest(state["optimizer"]), "scheduler_sha256": state_digest(state["scheduler"])}
            model.to(device)
            gpu_shadow = {key: value.to(device) for key, value in shadow.items()}
            shadow_device_sha = state_digest(gpu_shadow)
            ledger.update(initial=initial, checkpoint_sha256=ref["checkpoint"]["sha256"],
                          config_sha256=ref["effective_config_sha256"], steps=state["global_step"])
        yield config, model, gpu_shadow
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        with guard.phase(f"release/{label}", estimate_seconds=RELEASE_RESERVE, cleanup=True):
            try:
                ledger["checkpoint_unchanged"] = file_sha(path) == ref["checkpoint"]["sha256"]
                if not ledger["checkpoint_unchanged"]:
                    raise RuntimeError("checkpoint file changed during read-only execution")
                if state is not None and "initial" in ledger and gpu_shadow is not None:
                    after = {"model_sha256": state_sha(model), "ema_sha256": state_digest(state["ema"]),
                             "optimizer_sha256": state_digest(state["optimizer"]),
                             "scheduler_sha256": state_digest(state["scheduler"])}
                    checks = {key: value == initial[key] for key, value in after.items()}
                    checks["ema_device_values"] = state_digest(gpu_shadow) == shadow_device_sha
                    ledger.update(after=after, invariance=checks)
                    if not all(checks.values()):
                        raise RuntimeError("checkpoint model, BN, EMA, optimizer or scheduler state changed")
            finally:
                if model is not None:
                    model.cpu()
                if gpu_shadow is not None:
                    gpu_shadow.clear()
                torch.cuda.empty_cache()


def synthetic_preflight(device, guard, ledger):
    for variant in ("shared_2", "unshared_2"):
        model = None
        with guard.phase(f"synthetic/{variant}", estimate_seconds=35 + RELEASE_RESERVE):
            try:
                with torch.random.fork_rng(devices=[device.index]):
                    torch.random.default_generator.manual_seed(70703)
                    model = LoopCrowd(variant).to(device)
                    images = torch.randn(2, 3, 512, 512)
                    sample = point_targets([[17, 21], [259, 257], [500, 495]], 512, 512)
                    target = default_collate([sample, sample])
                    result = immutable_readout(model, {}, images, target, "live", device)
                    del result["comparisons"]
                    ledger[variant] = result
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                del model
                torch.cuda.empty_cache()


def measure_plan(plan, manifest, records, device, guard, run, events):
    seen_batch = {}
    for ref in plan["checkpoints"]:
        seed, arm = ref["seed"], ref["arm"]
        checkpoint_id = f"{seed}/{arm}"
        metadata = run["checkpoints"].setdefault(checkpoint_id, {})
        batches = next(row["batches"] for row in plan["plans"] if row["seed"] == seed)
        try:
            with cuda_checkpoint(ref, plan, device, guard, metadata) as (config, model, shadow):
                crops = TrainingCrops(records, manifest.parent, config, CROP_EPOCH)
                positions = {row.image_id: index for index, row in enumerate(records)}
                for batch in batches:
                    images = target = None
                    for policy in POLICIES:
                        identity = seed, batch["index"]
                        meta = {"seed": seed, "arm": arm, "batch": batch["index"], "policy": policy,
                                "density_group": batch["density_group"], "batch_sha256": batch["batch_sha256"]}
                        last = [p["elapsed_seconds"] for p in guard.phases if p["name"].startswith(("case/", "synthetic/"))]
                        estimate = 2 * max(last, default=2.0) + RELEASE_RESERVE
                        observation = meta | {"status": "stopped"}
                        try:
                            with guard.phase(f"case/{checkpoint_id}/{batch['index']}/{policy}",
                                             estimate_seconds=estimate, batch_policy_case=True):
                                if images is None:
                                    images, target = sample_batch(crops, positions, batch)
                                actual = state_digest((images, target))
                                if seen_batch.setdefault(identity, actual) != actual:
                                    raise RuntimeError("transformed crop or labels changed across paired reads")
                                observation["batch_tensor_sha256"] = actual
                                observation["measurement"] = immutable_readout(model, shadow, images, target, policy, device)
                                torch.cuda.empty_cache()
                            observation["status"] = "completed"
                        except BaseException as error:
                            observation["error"] = f"{type(error).__name__}: {error}"
                            raise
                        finally:
                            events.write(json.dumps(observation, allow_nan=False) + "\n")
                            events.flush()
                        run["completed_cases"] += 1
        finally:
            # cuda_checkpoint has already moved the live model off device and cleared its EMA mapping.
            model = shadow = None
        print(f"Completed {checkpoint_id} · {run['completed_cases']} cases · {guard.seconds:.2f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-scope", choices=(APPROVAL,), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    run = {"status": "in_progress", "completed_cases": 0, "checkpoints": {}, "synthetic_preflight": {},
           "limits": PILOT_CAP, "probe_manifest_sha256": PROBE_SHA, "owner_approval_scope": args.approved_scope,
           "project_optimizer_updates": 0, "external_experiment_download_bytes": 0,
           "python": platform.python_version(), "torch": str(torch.__version__), "command": [sys.executable, *sys.argv]}
    guard = None
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", DRAFT, "HEAD"], check=True)
        subprocess.run(["git", "diff", "HEAD", "--quiet"], check=True)
        run["execution_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        plan, manifest, records, weight = verified_inputs()
        device = torch.device("cuda:0")
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        guard = ProbeBudget(seconds=PILOT_CAP["gpu_seconds"], reserved_bytes=PILOT_CAP["process_peak_reserved_bytes"],
                            cases=PILOT_CAP["cases"], synchronize=lambda: torch.cuda.synchronize(device),
                            peak_reserved=lambda: torch.cuda.max_memory_reserved(device))
        with guard.phase("startup", estimate_seconds=20 + RELEASE_RESERVE):
            name = torch.cuda.get_device_name(device)
            if torch.cuda.device_count() != 1 or "3090" not in name or not torch.cuda.is_bf16_supported():
                raise RuntimeError("approved execution requires one visible BF16-capable RTX 3090")
            free, total = torch.cuda.mem_get_info(device)
            run.update(gpu=name, cuda_runtime=torch.version.cuda, global_gpu_bytes_free_before=free, gpu_bytes_total=total)
            if free < PILOT_CAP["process_peak_reserved_bytes"]:
                raise RuntimeError("system-wide free memory is below the process allocation cap")
        with (args.output / "cases.jsonl").open("x", encoding="utf-8") as events:
            synthetic_preflight(device, guard, run["synthetic_preflight"])
            print(f"GPU preflight passed · {guard.seconds:.2f} s · {guard.peak_bytes} bytes", flush=True)
            measure_plan(plan, manifest, records, device, guard, run, events)
        if file_sha(manifest) != MANIFEST_SHA or file_sha(weight) != WEIGHT_SHA:
            raise RuntimeError("manifest or initialization cache changed during execution")
        if run["completed_cases"] != 96 or guard.cases != 96:
            raise RuntimeError("frozen probe denominator differs")
        run["status"] = "completed"
    except BaseException as error:
        run.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if guard is not None:
            run.update(gpu_seconds=guard.seconds, peak_reserved_bytes=guard.peak_bytes,
                       entered_cases=guard.cases, phases=guard.phases, refusals=guard.refusals)
        write_json(args.output / "execution.json", run)


if __name__ == "__main__":
    main()
