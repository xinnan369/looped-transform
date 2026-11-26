"""Verify shifted-window parity, frozen B references and CPU resource projections."""

import argparse
import json
from pathlib import Path

import torch

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, make_model, seed_all, write_json
from loopcrowd.manifest import read_manifest
from scripts.prepare_lc007d import initialization, planned_trajectory
from scripts.prepare_lc007e import SCREEN as K4_SCREEN, config_for, trajectory_columns
from scripts.prepare_lc007f import ARCHIVE as K4_ARCHIVE, history
from scripts.probe_cross_window import TOLERANCE, probe
from scripts.profile import forward_costs
from scripts.run_lc007a import read_json
from scripts.run_lc007b import frozen_inputs

SCREEN = Path("configs/lc007g_cross_window.json")


def project(costs, phases):
    fixed, shifted = (costs[arm] for arm in ("U4", "U4-shift"))
    train_ratio = shifted["midpoint_final"]["dense_flops"] / fixed["midpoint_final"]["dense_flops"]
    eval_ratio = shifted["four_readouts"]["dense_flops"] / fixed["four_readouts"]["dense_flops"]
    relevant = [row for row in phases if row["case"].split("/")[-1] in ("S4", "U4")]
    totals = {phase: sum(row["seconds"] for row in relevant if row["phase"] == phase)
              for phase in ("training", "validation_live", "validation_ema", "checkpoint")}
    projection = {"training": totals["training"] * train_ratio, "checkpoint": totals["checkpoint"],
                  **{phase: totals[phase] * eval_ratio for phase in ("validation_live", "validation_ema")}}
    overhead = 120
    central = sum(projection.values()) + overhead
    peak = max(row["peak_reserved_bytes"] for row in relevant)
    return {"historical_matched_phase_seconds": totals, "training_compute_ratio": train_ratio,
            "validation_compute_ratio": eval_ratio, "projected_phase_seconds": projection,
            "setup_transfer_cleanup_reserve_seconds": overhead, "central_seconds": central,
            "upper_with_20pct_contingency_seconds": central * 1.2,
            "historical_matched_peak_reserved_bytes": peak, "padded_window_ratio": 81 / 64,
            "required_device_headroom_bytes_estimate": int(peak * 81 / 64 + 64 * 1024**2),
            "memory_policy": "observe_peak_and_device_free_no_process_quota", "actual_gpu": "not_run"}


def architecture_costs(model, tile):
    calls, windows = [], []
    handles = []
    for index, block in enumerate(model.blocks):
        handles.append(block.register_forward_pre_hook(
            lambda module, args, index=index: calls.append({"block_index": index, "shift": args[1]})))
        handles.append(block.attn.register_forward_pre_hook(
            lambda module, args: windows.append(list(args[0].shape))))
    try:
        final = forward_costs(model, tile)
    finally:
        for handle in handles:
            handle.remove()
    for entry, (groups, tokens, channels) in zip(calls, windows, strict=True):
        entry.update(windows=groups, window_tokens=tokens, channels=channels)
    return {"final_only": final, "midpoint_final": forward_costs(model, tile, depth_midpoint=True),
            "four_readouts": forward_costs(model, tile, depth_staged=True)}, calls


def audit(manifest, screen, *, mechanism=True):
    tables = history()
    _, data = frozen_inputs(manifest)
    previous = read_json(K4_SCREEN)
    prior = read_json(K4_ARCHIVE / "initialization_audit.json")
    for key in ("manifest_sha256", "pretrained_weight_sha256", "ordered_val_ids", "train_only_density_tertiles"):
        if prior["data"][key] != data[key]:
            raise ValueError(f"cached input differs at {key}")
    records = read_manifest(manifest, splits=("train", "val"))
    if any(not (manifest.parent / row.image_path).is_file() for row in records):
        raise FileNotFoundError("cached B images are incomplete")
    train = [row for row in records if row.split == "train"]
    starts, costs, geometry = {}, {}, {}
    tile = torch.zeros(1, 3, 512, 512)
    for seed_text, arms in screen["order"].items():
        seed = int(seed_text)
        for arm in arms:
            config = config_for(screen, seed, arm)
            reference = screen["arms"][arm]["comparison"]
            baseline = config_for(previous, seed, reference)
            old = prior["planned_starts"][f"{seed}/{reference}"]
            if (baseline["variant"] != screen["fixed_references"][reference]
                    or json_digest(baseline) != old["config_sha256"]
                    or config != baseline | {"variant": screen["arms"][arm]["variant"]}):
                raise ValueError(f"matching recipe differs at {seed}/{arm}")
            if list(planned_trajectory(train, manifest.parent, config)) != trajectory_columns(tables[seed, reference, "training"]):
                raise ValueError(f"historical crop/LR trajectory differs at {seed}/{arm}")
            seed_all(seed)
            model = make_model(config, initialize=True)
            state = initialization(model)
            if (any(state[key] != old[key] for key in (
                    "initial_sha256", "common_sha256", "decoder_sha256", "block_sha256", "rng_after_init_sha256"))
                    or model.channel_gate_deltas is not None or model.embedding is not None):
                raise ValueError(f"full-state or independent-block initialization differs at {seed}/{arm}")
            starts[f"{seed}/{arm}"] = state | {"reference": reference, "config_sha256": json_digest(config)}
            if seed == 42:
                for name, target in ((arm, model), (reference, make_model(baseline, initialize=False))):
                    target.eval()
                    costs[name], geometry[name] = architecture_costs(target, tile)
    if costs["S4"] != costs["U4"] or costs["S4-shift"] != costs["U4-shift"]:
        raise ValueError("matched sharing FLOPs differ")
    phases = [json.loads(line) for line in (K4_ARCHIVE / "gpu_phases.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"status": "cpu_audit_pass", "data": data, "historical_tables_verified": len(tables),
            "plan_sha256": file_sha(SCREEN), "arm_config_sha256": file_sha(Path(screen["arm_config"])),
            "planned_starts": starts, "forward_costs": costs, "resource_projection": project(costs, phases),
            "token_mechanism": {key: {"atol": tol, "results": probe(key)} for key, tol in TOLERANCE.items()} if mechanism else None,
            "observed_call_geometry": geometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.manifest, read_json(SCREEN)))


if __name__ == "__main__":
    main()
