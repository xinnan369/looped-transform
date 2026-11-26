"""Recompute inputs, historical trajectories and resource projections on CPU."""

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, make_model, read_config, seed_all, write_json
from loopcrowd.manifest import read_manifest
from scripts.archive_lc007c import check_training, load_readouts, read_frozen_files
from scripts.prepare_lc007d import initialization, planned_trajectory
from scripts.profile import forward_costs
from scripts.run_lc007a import read_csv
from scripts.run_lc007b import frozen_inputs

SCREEN = Path("configs/lc007e_depth_screen.json")
B = Path("reports/LC-007B")
D = Path("reports/LC-007D")


def config_for(screen, seed, arm):
    return read_config(Path(screen["arm_config"]), screen["arms"][arm]["variant"]) | {"seed": seed}


def history():
    b = read_frozen_files(B)
    load_readouts(b, B)
    d = {}
    for row in read_csv(D / "epoch_index.csv"):
        key = int(row["seed"]), row["arm"], row["kind"]
        path = D / "epochs" / row["seed"] / row["arm"] / f"{row['kind']}.csv"
        if (key in d or path != D / row["path"] or row["status"] != "completed"
                or file_sha(path) != row["sha256"] or path.stat().st_size != int(row["bytes"])):
            raise ValueError(f"LC-007D indexed evidence differs at {key}")
        d[key] = read_csv(path)
        if len(d[key]) != int(row["rows"]):
            raise ValueError(f"LC-007D row count differs at {key}")
    expected = {(s, a, k) for s in (42, 123) for a in ("R", "RS", "U-RS") for k in ("training", "metrics", "images")}
    if set(d) != expected:
        raise ValueError("LC-007D index coverage differs")
    for s in (42, 123):
        for a in ("R", "RS", "U-RS"):
            check_training(d[s, a, "training"])
            reference = "unshared_2_aux_025" if a == "U-RS" else "shared_2_aux_025"
            if trajectory_columns(d[s, a, "training"]) != trajectory_columns(b[s, reference, "training"]):
                raise ValueError(f"LC-007D actual pairing differs at {s}/{a}")
    return b | d


def trajectory_columns(rows):
    return [(row["crop_order_sha256"], row["learning_rates_sha256"]) for row in rows]


def resource_projection(costs):
    phases = [json.loads(line) for line in (D / "gpu_phases.jsonl").read_text(encoding="utf-8").splitlines()]
    d_final = costs["K2"]["final_only"]["dense_flops"]
    d_staged = costs["K2"]["staged"]["dense_flops"]
    k4_final = costs["U4RS"]["final_only"]["dense_flops"]
    k4_staged = costs["U4RS"]["staged"]["dense_flops"]
    # The difference between K2 staged and final is one point decoder.
    decoder = d_staged - d_final
    train_ratio = (k4_final + decoder) / d_staged
    eval_ratio = k4_staged / d_staged
    projected = {}
    for phase, ratio in (("training", train_ratio), ("validation_live", eval_ratio),
                         ("validation_ema", eval_ratio), ("checkpoint", 2.0)):
        projected[phase] = 320 * mean(row["seconds"] for row in phases if row["phase"] == phase) * ratio
    remaining = sum(projected.values()) + 90
    peak = max(row["peak_reserved_bytes"] for row in phases)
    return {"historic_seconds_for_six": phases[-1]["charged_seconds"], "historic_peak_reserved": peak,
            "train_compute_ratio": train_ratio, "eval_compute_ratio": eval_ratio,
            "projected_phase_seconds": projected, "preflight_setup_cleanup_reserve_seconds": 90,
            "central_seconds": remaining, "upper_with_15pct_contingency_seconds": remaining * 1.15,
            "peak_reserved_scenario_2x_bytes": 2 * peak, "actual_gpu": "not_run"}


def audit(manifest, screen, *, profile=True):
    table = history()
    frozen, data = frozen_inputs(manifest)
    archived = json.loads((D / "initialization_audit.json").read_text(encoding="utf-8"))
    for key in ("manifest_sha256", "pretrained_weight_sha256", "ordered_val_ids", "train_only_density_tertiles"):
        if archived["data"][key] != data[key]:
            raise ValueError(f"LC-007D cached input differs for {key}")
    records = read_manifest(manifest, splits=("train", "val"))
    if any(not (manifest.parent / row.image_path).is_file() for row in records):
        raise FileNotFoundError("cached B images are incomplete")
    train = [row for row in records if row.split == "train"]
    starts, costs = {}, {}
    for seed_text, order in screen["order"].items():
        seed = int(seed_text)
        structure = {}
        for arm in order:
            config = config_for(screen, seed, arm)
            reference = screen["arms"][arm]["comparison"]
            matched = archived["planned_gated_initialization" if arm.endswith("RS") else "historical_initialization"]
            old = matched[f"{seed}/{reference}"]
            baseline_arm = (matched[f"{seed}/{reference}"]["reference"] if arm.endswith("RS") else reference)
            baseline = read_config(Path(frozen["configs"][baseline_arm])) | {"seed": seed, "ema_decay": .99}
            historical_variant = {"RS": "shared_2_channel_residual_state",
                                  "U-RS": "unshared_2_channel_residual_state"}.get(reference, baseline["variant"])
            if json_digest(baseline | {"variant": historical_variant}) != old["config_sha256"]:
                raise ValueError(f"archived reference config digest differs for {seed}/{reference}")
            if {k: v for k, v in config.items() if k not in ("variant", "depth_screen")} != {
                k: v for k, v in baseline.items() if k != "variant"
            }:
                raise ValueError(f"recipe mismatch for {seed}/{arm}")
            if list(planned_trajectory(train, manifest.parent, config)) != trajectory_columns(table[seed, reference, "training"]):
                raise ValueError(f"trajectory mismatch for {seed}/{arm}")
            seed_all(seed)
            model = make_model(config, initialize=True)
            hashes = initialization(model)
            if (hashes["common_sha256"] != old["common_sha256"]
                    or hashes["block_sha256"][:len(old["block_sha256"])] != old["block_sha256"]
                    or len(set(hashes["block_sha256"])) != (1 if model.shared else 4)):
                raise ValueError(f"initialization mismatch for {seed}/{arm}")
            gate = model.channel_gate_deltas
            if arm.endswith("RS") and (gate.numel() != 2304 or gate.dtype != torch.float32 or torch.count_nonzero(gate)):
                raise ValueError(f"gate initialization mismatch for {seed}/{arm}")
            starts[f"{seed}/{arm}"] = hashes | {"config_sha256": json_digest(config), "reference": reference}
            structure[arm] = hashes
            if profile and seed == 42:
                model.eval()
                tile = torch.zeros(1, 3, 512, 512)
                costs[arm] = {"final_only": forward_costs(model, tile),
                              "staged": forward_costs(model, tile, depth_staged=True)}
        for ungated, gated in (("S4", "S4RS"), ("U4", "U4RS")):
            for key in ("common_sha256", "block_sha256", "rng_after_init_sha256"):
                if structure[ungated][key] != structure[gated][key]:
                    raise ValueError(f"factorial initialization mismatch for {seed}/{gated}")
    if profile:
        # Add one decoder invocation to the K2 final-only estimate.
        model = make_model(read_config(Path(frozen["configs"]["shared_2_aux_025"])), initialize=False).eval()
        tile = torch.zeros(1, 3, 512, 512)
        final = forward_costs(model, tile)
        decoder = costs["S4"]["staged"]["dense_flops"] - costs["S4"]["final_only"]["dense_flops"]
        costs["K2"] = {"final_only": final, "staged": final | {"dense_flops": final["dense_flops"] + decoder // 3}}
    return {"status": "cpu_audit_pass", "data": data, "historical_tables_verified": len(table),
            "plan_sha256": file_sha(SCREEN), "arm_config_sha256": file_sha(Path(screen["arm_config"])),
            "planned_starts": starts, "forward_costs": costs, "resource_projection": resource_projection(costs) if profile else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.manifest, json.loads(SCREEN.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
