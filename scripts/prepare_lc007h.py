"""Verify local inputs and freeze confidence refinement feasibility on CPU."""

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, make_model, seed_all, write_json
from loopcrowd.manifest import read_manifest
from loopcrowd.model.refinement import refinement_gain
from scripts.diagnose_lc007h import G, archives
from scripts.prepare_lc007d import initialization, planned_trajectory
from scripts.prepare_lc007e import config_for, trajectory_columns
from scripts.prepare_lc007g import SCREEN as SHIFT_SCREEN
from scripts.profile import forward_costs
from scripts.run_lc007a import read_json
from scripts.run_lc007b import frozen_inputs

SCREEN = Path("configs/lc007h_confidence_refinement.json")


def audit(manifest, proposal):
    tables = archives()
    _, data = frozen_inputs(manifest)
    refs = read_json(G / "initialization_audit.json")
    shifts = read_json(SHIFT_SCREEN)
    evidence = read_json(G / "run_evidence.json")
    for key in ("manifest_sha256", "pretrained_weight_sha256", "ordered_val_ids", "train_only_density_tertiles"):
        if data[key] != refs["data"][key]:
            raise ValueError(f"frozen shifted input differs at {key}")
    cases = {}
    for case in evidence["cases"]:
        checkpoint, run = (case[key] for key in ("retained_checkpoint", "manifest"))
        for item in (checkpoint, run):
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
                raise ValueError(f"local frozen artifact does not match {path}")
        provenance = read_json(Path(run["path"]))
        label = f"{case['seed']}/{case['arm']}"
        if (case["epochs"] != 40 or case["successful_updates"] != 7200
                or provenance["execution"] != "completed" or provenance["completed_epochs"] != 40
                or provenance["config_sha256"] != refs["planned_starts"][label]["config_sha256"]
                or provenance["data"]["sha256"] != data["manifest_sha256"]):
            raise ValueError(f"checkpoint provenance differs at {label}")
        cases[label] = checkpoint | {"config_sha256": provenance["config_sha256"], "epoch": 40, "ema_step": 7200}
    if len(cases) != 4:
        raise ValueError("the diagnostic requires exactly four E40 checkpoints")
    records = read_manifest(manifest, splits=("train", "val"))
    if any(not (manifest.parent / row.image_path).is_file() for row in records):
        raise FileNotFoundError("cached B images are incomplete")
    records = [row for row in records if row.split == "train"]
    initials, costs = {}, {}
    image = torch.zeros(1, 3, 512, 512)
    for seed_text, arms in proposal["order"].items():
        seed = int(seed_text)
        for arm in arms:
            spec = proposal["arms"][arm]
            ref = spec["reference"]
            config = config_for(shifts, seed, ref) | {"refinement": spec["refinement"]}
            historical = refs["planned_starts"][f"{seed}/{ref}"]
            if json_digest({k: v for k, v in config.items() if k != "refinement"}) != historical["config_sha256"]:
                raise ValueError(f"historical training recipe differs for {seed}/{arm}")
            seed_all(seed)
            model = make_model(config, initialize=True)
            init = initialization(model)
            parity = ("initial_sha256", "encoder_sha256", "decoder_sha256", "block_sha256",
                      "rng_after_init_sha256", "parameters")
            if any(init[key] != historical[key] for key in parity):
                raise ValueError(f"initialization differs for {seed}/{arm}")
            if list(planned_trajectory(records, manifest.parent, config)) != trajectory_columns(tables[seed, ref, "training"]):
                raise ValueError(f"crop/LR pairing differs for {seed}/{arm}")
            initials[f"{seed}/{arm}"] = init | {"config_sha256": json_digest(config), "reference": ref}
            if seed == 42:
                model.eval()
                costs[arm] = {"deployed": forward_costs(model, image),
                              "midpoint_final": forward_costs(model, image, depth_midpoint=True),
                              "four_readouts": forward_costs(model, image, depth_staged=True)}
    logits = torch.linspace(-4, 8, 48).reshape(1, 2, 4, 6)
    a, b = (refinement_gain(logits, mode=mode).flatten() for mode in ("guided", "placebo"))
    if torch.equal(a, b) or not torch.equal(a.sort().values, b.sort().values):
        raise ValueError("synthetic placebo must alter spatial alignment while preserving gains")
    phases = [json.loads(line) for line in (G / "gpu_phases.jsonl").read_text(encoding="utf-8").splitlines()]
    ema = [row["seconds"] for row in phases if row["phase"] == "validation_ema"]
    total = evidence["execution"]["charged_seconds"]
    return {"status": "cpu_audit_pass", "data": data, "historical_tables_verified": len(tables),
            "proposal_sha256": file_sha(SCREEN), "historical_execution_commit": evidence["execution"]["code_commit"],
            "diagnostic_checkpoints": cases, "checkpoint_inference": "not_run", "planned_initializations": initials,
            "dense_forward_costs": costs,
            "synthetic_placebo_alignment": {"mean": float(a.mean()), "correlation": float(torch.corrcoef(torch.stack((a, b)))[0, 1]),
                                            "different_cells": int(torch.count_nonzero(a != b)), "cells": a.numel()},
            "resource_proxy": {"historical_elapsed_seconds": total, "historical_ema_sweep_mean_seconds": mean(ema),
                               "historical_ema_sweep_max_seconds": max(ema),
                               "conditional_training_six_run_proxy_seconds": total * 1.5,
                               "conditional_training_with_25pct_allowance_and_120s_seconds": total * 1.5 * 1.25 + 120,
                               "actual_gpu": "not_run"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.manifest, read_json(SCREEN)))


if __name__ == "__main__":
    main()
