"""Verify the K=8 recipe, archived references and resource projections on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, make_model, seed_all, write_json
from loopcrowd.manifest import read_manifest
from loopcrowd.targets import point_targets
from loopcrowd.training import supervised_point_loss
from scripts.archive_lc007c import check_training
from scripts.prepare_lc007d import initialization, planned_trajectory
from scripts.prepare_lc007e import SCREEN as K4_SCREEN, config_for, history as earlier_history, trajectory_columns
from scripts.profile import forward_costs
from scripts.run_lc007a import read_csv, read_json
from scripts.run_lc007b import frozen_inputs

SCREEN = Path("configs/lc007f_depth_screen.json")
ARCHIVE = Path("reports/LC-007E")


def history():
    prior = earlier_history()
    screen = read_json(K4_SCREEN)
    archived = {}
    for row in read_csv(ARCHIVE / "epoch_index.csv"):
        seed, arm, kind = int(row["seed"]), row["arm"], row["kind"]
        path = ARCHIVE / "epochs" / str(seed) / arm / f"{kind}.csv"
        key = seed, arm, kind
        if (key in archived or path != ARCHIVE / row["path"] or row["status"] != "completed"
                or path.stat().st_size != int(row["bytes"]) or file_sha(path) != row["sha256"]):
            raise ValueError(f"LC-007E indexed file differs at {key}")
        archived[key] = read_csv(path)
        if len(archived[key]) != int(row["rows"]):
            raise ValueError(f"LC-007E row count differs at {key}")
    expected = {(int(seed), arm, kind) for seed, arms in screen["order"].items() for arm in arms
                for kind in ("training", "metrics", "images")}
    if set(archived) != expected:
        raise ValueError("LC-007E archive coverage differs")
    for seed, arms in screen["order"].items():
        for arm in arms:
            rows = archived[int(seed), arm, "training"]
            reference = screen["arms"][arm]["comparison"]
            check_training(rows)
            if trajectory_columns(rows) != trajectory_columns(prior[int(seed), reference, "training"]):
                raise ValueError(f"depth reference trajectory differs at {seed}/{arm}")
    return prior | archived


def saved_storage_bytes(config):
    """Sum unique storages retained by the BF16 CPU midpoint/final loss graph."""
    model = make_model(config, initialize=False).train()
    images = torch.zeros(2, 3, 512, 512)
    target = {key: torch.stack((value, value)) for key, value in
              point_targets(np.array([[128.5, 128.5]]), 512, 512, 2).items()}
    storages = {}

    def retain(tensor):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(retain, lambda tensor: tensor), torch.autocast("cpu", torch.bfloat16):
        loss = supervised_point_loss(model, images, target, config)
    if not torch.isfinite(loss):
        raise ValueError("CPU BF16 synthetic loss is nonfinite")
    return sum(storage.nbytes() for storage in storages.values())


def project(screen, costs, rows, old_storage, new_storage, added_parameters):
    totals = {phase: sum(row["seconds"] for row in rows if row["phase"] == phase)
              for phase in ("training", "validation_live", "validation_ema", "checkpoint")}
    old, new = (costs[arm] for arm in ("U4RS", "U8RS"))
    train = new["midpoint_final"]["dense_flops"] / old["midpoint_final"]["dense_flops"]
    early = new["midpoint_final"]["dense_flops"] / old["all_readouts"]["dense_flops"]
    late = new["all_readouts"]["dense_flops"] / old["all_readouts"]["dense_flops"]
    start = config_for(screen, 42, "U8RS")["depth_readout_start_epoch"]
    early_epochs = start - 1
    ratio = (early_epochs * early + (40 - early_epochs) * late) / 40
    projected = {"training": totals["training"] * train, "checkpoint": totals["checkpoint"] * 2,
                 **{name: totals[name] * ratio for name in ("validation_live", "validation_ema")}}
    central = sum(projected.values()) + 120
    historic_peak = max(row["peak_reserved_bytes"] for row in rows)
    memory = int(historic_peak * (new_storage / old_storage) + 20 * added_parameters + 64 * 1024**2)
    return {"historical_phase_seconds": totals, "training_compute_ratio": train,
            "early_validation_ratio": early, "late_validation_ratio": late, "weighted_validation_ratio": ratio,
            "projected_phase_seconds": projected, "preflight_transfers_cleanup_reserve_seconds": 120,
            "central_seconds": central, "upper_with_20pct_contingency_seconds": central * 1.20,
            "historical_peak_reserved_bytes": historic_peak, "cpu_bf16_saved_storage_bytes": {"K4": old_storage, "K8": new_storage},
            "added_unshared_parameter_bytes_proxy": 20 * added_parameters,
            "peak_reserved_projection_bytes": memory, "actual_gpu": "not_run"}


def audit(manifest, screen):
    table = history()
    _, data = frozen_inputs(manifest)
    prior = read_json(ARCHIVE / "initialization_audit.json")
    previous = read_json(K4_SCREEN)
    for key in ("manifest_sha256", "pretrained_weight_sha256", "ordered_val_ids", "train_only_density_tertiles"):
        if prior["data"][key] != data[key]:
            raise ValueError(f"cached input differs at {key}")
    records = read_manifest(manifest, splits=("train", "val"))
    if any(not (manifest.parent / row.image_path).is_file() for row in records):
        raise FileNotFoundError("cached B images are incomplete")
    train = [row for row in records if row.split == "train"]
    starts, costs = {}, {}
    tile = torch.zeros(1, 3, 512, 512)
    for seed_text, order in screen["order"].items():
        seed = int(seed_text)
        for arm in order:
            config = config_for(screen, seed, arm)
            spec = screen["arms"][arm]
            reference = spec["comparison"]
            baseline = config_for(previous, seed, reference)
            old = prior["planned_starts"][f"{seed}/{reference}"]
            if (config["depth_readout_start_epoch"] != 36
                    or json_digest(baseline) != old["config_sha256"]
                    or {k: v for k, v in config.items() if k != "depth_readout_start_epoch"}
                    != (baseline | {"variant": spec["variant"]})):
                raise ValueError(f"reference recipe differs at {seed}/{arm}")
            trajectory = list(planned_trajectory(train, manifest.parent, config))
            if any(trajectory != trajectory_columns(table[seed, ref, "training"]) for ref in (reference, spec["depth2"])):
                raise ValueError(f"historical trajectory differs at {seed}/{arm}")
            seed_all(seed)
            model = make_model(config, initialize=True)
            state = initialization(model)
            blocks = state["block_sha256"]
            if (state["common_sha256"] != old["common_sha256"] or blocks[:len(old["block_sha256"])] != old["block_sha256"]
                    or len(set(blocks)) != (1 if model.shared else 8)):
                raise ValueError(f"reference initialization differs at {seed}/{arm}")
            gate = model.channel_gate_deltas
            if arm.endswith("RS") and (gate.numel() != 4608 or gate.dtype != torch.float32 or torch.count_nonzero(gate)):
                raise ValueError(f"gate initialization differs at {seed}/{arm}")
            starts[f"{seed}/{arm}"] = state | {"config_sha256": json_digest(config),
                                               "reference": reference, "reference_depth2": spec["depth2"]}
            if seed == 42:
                model.eval()
                costs[arm] = {"final_only": forward_costs(model, tile),
                              "midpoint_final": forward_costs(model, tile, depth_midpoint=True),
                              "all_readouts": forward_costs(model, tile, depth_staged=True)}
        for ungated, gated in (("S8", "S8RS"), ("U8", "U8RS")):
            a, b = (starts[f"{seed}/{arm}"] for arm in (ungated, gated))
            if b["parameters"] - a["parameters"] != 4608 or any(a[key] != b[key] for key in
                    ("common_sha256", "block_sha256", "rng_after_init_sha256")):
                raise ValueError(f"factorial initialization differs for {seed}/{gated}")
    old_config = config_for(previous, 42, "U4RS")
    old_model = make_model(old_config, initialize=False).eval()
    costs["U4RS"] = {"midpoint_final": forward_costs(old_model, tile, depth_midpoint=True),
                    "all_readouts": forward_costs(old_model, tile, depth_staged=True)}
    ledger = [json.loads(line) for line in (ARCHIVE / "gpu_phases.jsonl").read_text(encoding="utf-8").splitlines()]
    projection = project(screen, costs, ledger, saved_storage_bytes(old_config),
                         saved_storage_bytes(config_for(screen, 42, "U8RS")),
                         starts["42/U8RS"]["parameters"] - prior["planned_starts"]["42/U4RS"]["parameters"])
    return {"status": "cpu_audit_pass", "data": data, "historical_tables_verified": len(table),
            "plan_sha256": file_sha(SCREEN), "arm_config_sha256": file_sha(Path(screen["arm_config"])),
            "planned_starts": starts, "forward_costs": costs, "resource_projection": projection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.manifest, read_json(SCREEN)))


if __name__ == "__main__":
    main()
