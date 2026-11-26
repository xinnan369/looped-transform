"""Audit the LC-007D screen on CPU using the existing manifest and weight cache."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from loopcrowd.calibration import json_digest
from loopcrowd.data import TrainingCrops
from loopcrowd.experiment import file_sha, make_model, read_config, seed_all, state_sha, write_json
from loopcrowd.manifest import read_manifest
from loopcrowd.scheduling import UpdateSchedule
from scripts.archive_lc007c import ARMS, UNSHARED, load_readouts, read_frozen_files
from scripts.profile import forward_costs
from scripts.run_lc007b import COMMON, frozen_inputs

CONFIG = Path("configs/lc007d_channel_screen.json")
ENCODER = ("p2.", "p3.", "project.")
DECODER = ("fuse.", "point_logits.", "point_offsets.")
FROZEN_EVAL_SOURCES = (
    "accuracy_diagnosis.py", "calibration.py", "crops.py", "ema.py", "grouped_validation.py",
    "inference.py", "manifest.py", "metrics.py", "scheduling.py", "targets.py",
)


def initialization(model):
    return {
        "initial_sha256": state_sha(model), "common_sha256": state_sha(model, prefixes=COMMON),
        "encoder_sha256": state_sha(model, prefixes=ENCODER), "decoder_sha256": state_sha(model, prefixes=DECODER),
        "block_sha256": [state_sha(block) for block in model.blocks],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "rng_after_init_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def planned_trajectory(records, directory, config):
    steps = (len(records) + config["batch_size"] - 1) // config["batch_size"]
    schedule = UpdateSchedule(SimpleNamespace(param_groups=[{}]), config, steps * config["epochs"])
    if steps != 180 or schedule.warmup_updates != 360:
        raise ValueError("LC-007D schedule dimensions differ")
    for epoch in range(1, config["epochs"] + 1):
        crops = TrainingCrops(records, directory, config, epoch)
        # Workers load samples; the same parent-process sampler picks their order.
        sampler = DataLoader(range(len(crops)), batch_size=config["batch_size"], shuffle=True,
                             generator=torch.Generator().manual_seed(config["seed"] + epoch))
        order = hashlib.sha256()
        rates = []
        for batch in sampler:
            identities = []
            for index in batch.tolist():
                record = records[index]
                points = np.asarray(record.points, dtype=float).reshape(-1, 2)
                identities.append((index, *crops.origin(record, points)))
            order.update(np.asarray(identities, dtype="<i8").tobytes())
            rates.append(schedule.apply())
            schedule.global_step += 1
        yield order.hexdigest(), json_digest(rates)


@torch.inference_mode()
def synthetic_profile(model):
    model.eval()
    image = torch.zeros(1, 3, 512, 512)
    costs = forward_costs(model, image)
    times = []
    for _ in range(5):
        start = perf_counter()
        model(image)
        times.append((perf_counter() - start) * 1000)
    return costs | {"synthetic_cpu_fp32_batch1_ms": times, "median_ms": median(times)}


def audit(manifest, screen, *, profile=True):
    archive = Path(screen["historical_archive"])
    tables = read_frozen_files(archive)
    load_readouts(tables, archive)
    historic = json.loads((archive / "preflight.json").read_text(encoding="utf-8"))
    revision = historic["provenance"]["git_commit"]
    subprocess.run(["git", "diff", "--quiet", revision, "--",
                    *(f"src/loopcrowd/{source}" for source in FROZEN_EVAL_SOURCES)], check=True)
    baseline, data = frozen_inputs(manifest)
    if baseline != historic["frozen"] or data["ordered_val_ids"] != historic["ordered_val_ids"]:
        raise ValueError("historical fixture or validation order differs")
    if screen["ema_decay"] != baseline["ema_decay"]:
        raise ValueError("EMA decay differs")
    train = read_manifest(manifest, splits=("train",))
    old, new, profiles = {}, {}, {}
    missing = [record.image_id for record in read_manifest(manifest, splits=("train", "val"))
               if not (manifest.parent / record.image_path).is_file()]
    if missing:
        raise FileNotFoundError(f"cached B images missing: {len(missing)}; first missing ID {missing[0]}")
    for seed_text, order in screen["order"].items():
        seed = int(seed_text)
        for arm in ARMS:
            path = Path(baseline["configs"][arm])
            saved = historic["cases"][f"{seed}/{arm}"]
            config = read_config(path) | {"seed": seed, "ema_decay": screen["ema_decay"]}
            if file_sha(path) != historic["arm_configs"][arm]["sha256"] or json_digest(config) != saved["config_sha256"]:
                raise ValueError(f"historical config differs for {seed}/{arm}")
            seed_all(seed)
            model = make_model(config, initialize=True)
            hashes = initialization(model)
            if any(hashes[key] != saved[key] for key in ("initial_sha256", "common_sha256", "block_sha256", "parameters")):
                raise ValueError(f"historical initialization differs for {seed}/{arm}")
            trajectory = list(planned_trajectory(train, manifest.parent, config))
            archived = [(row["crop_order_sha256"], row["learning_rates_sha256"])
                        for row in tables[seed, arm, "training"]]
            if trajectory != archived:
                raise ValueError(f"planned crop or LR schedule differs for {seed}/{arm}")
            old[f"{seed}/{arm}"] = hashes | {"config_sha256": saved["config_sha256"], "epochs_matched": len(trajectory)}
            if profile and seed == 42:
                profiles[arm] = synthetic_profile(model)
        for arm in order:
            spec = screen["variants"][arm]
            reference = spec["reference"]
            config = read_config(Path(baseline["configs"][reference]), spec["variant"]) | {
                "seed": seed, "ema_decay": screen["ema_decay"],
            }
            seed_all(seed)
            model = make_model(config, initialize=True)
            hashes = initialization(model)
            previous = old[f"{seed}/{reference}"]
            keys = ("common_sha256", "encoder_sha256", "decoder_sha256", "block_sha256", "rng_after_init_sha256")
            gates = model.channel_gate_deltas
            if (any(hashes[key] != previous[key] for key in keys)
                    or hashes["parameters"] - previous["parameters"] != spec["gate_parameters"]
                    or gates.dtype != torch.float32 or torch.count_nonzero(gates)):
                raise ValueError(f"gated initialization differs for {seed}/{arm}")
            new[f"{seed}/{arm}"] = hashes | {
                "config_sha256": json_digest(config), "reference": reference,
                "gate_parameters": gates.numel(), "actual_run_crop_and_lr": "not_run",
            }
            if profile and seed == 42:
                profiles[arm] = synthetic_profile(model)
        blocks = [old[f"{seed}/{arm}"]["block_sha256"] for arm in ARMS if arm != "cnn_only"]
        if len({value[0] for value in blocks}) != 1 or len(set(old[f"{seed}/{UNSHARED}"]["block_sha256"])) != 2:
            raise ValueError(f"first-block parity or independent second block differs for seed {seed}")
    return {
        "status": "cpu_audit_pass", "reference_files_verified": len(tables), "config_sha256": file_sha(CONFIG),
        "historical_code_commit": revision, "unchanged_evaluation_sources": list(FROZEN_EVAL_SOURCES),
        "data": data, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "threads": torch.get_num_threads(), "device": "cpu", "historical_initialization": old,
        "planned_gated_initialization": new, "synthetic_forward_profiles": profiles,
        "gpu_profile": "not_run", "reference_reuse_during_screen": "not_run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(args.output, audit(args.manifest, config))


if __name__ == "__main__":
    main()
