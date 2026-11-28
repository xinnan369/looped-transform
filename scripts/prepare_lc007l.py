"""Audit the one-arm loop-normalization freeze against archived native crops."""

from pathlib import Path

import torch

from loopcrowd.calibration import json_digest
from loopcrowd.ema import ParameterEMA
from loopcrowd.experiment import make_model, seed_all, state_sha
from loopcrowd.training import optimizer_for
from scripts.prepare_lc007d import initialization
from scripts.prepare_lc007i import audit as native_audit, recipe as native_recipe
from scripts.prepare_lc007k import SHARED, SHARED_ARM, UNSHARED_ARM, assert_paired_exposure, crop_control, shared_control
from scripts.profile import forward_costs
from scripts.run_lc007d import read_json

SCREEN = Path("configs/lc007l_loopnorm.json")
UNSHARED = Path("reports/LC-007K/training_results")
ARM = "S4-shift-768-LN"
CASE = f"42/{ARM}"


def recipe(plan):
    return native_recipe(plan) | {"loop_norm": plan["loop_norm"]}


def controls(plan):
    return (shared_control(plan), crop_control(UNSHARED, plan["u768_sha256"], UNSHARED_ARM))


def audit(manifest, plan):
    (s, s_evidence), (u, u_evidence) = controls(plan)
    evidence = native_audit(manifest, plan, screen=SCREEN)
    planned = [r for r in evidence["exposure"] if r["crop_size"] == plan["train_crop_size"]]
    for actual in (s["training"], u["training"]):
        assert_paired_exposure(planned, actual)
    reference = evidence["planned_starts"][CASE]
    for folder, arm in ((SHARED, SHARED_ARM), (UNSHARED, UNSHARED_ARM)):
        old = read_json(folder / "preflight_cpu.json")
        for key in ("manifest_sha256", "pretrained_weight_sha256", "ordered_val_ids", "train_only_density_tertiles"):
            if old["data"][key] != evidence["data"][key]:
                raise ValueError("paired native input provenance differs")
        if folder == SHARED and old["planned_starts"][f"42/{arm}"] != reference:
            raise ValueError("historical shared 768 initialization differs")
    seed_all(42)
    base = make_model(native_recipe(plan), initialize=True).eval()
    seed_all(42)
    config = recipe(plan)
    model = make_model(config, initialize=True).eval()
    start = initialization(model)
    if any(start[k] != reference[k] for k in ("common_sha256", "encoder_sha256", "decoder_sha256",
                                             "block_sha256", "rng_after_init_sha256")):
        raise ValueError("loop norm altered an archived initial tensor or RNG")
    base_state, state = base.state_dict(), model.state_dict()
    if any(not torch.equal(state[name], value) for name, value in base_state.items()):
        raise ValueError("a base state tensor differs")
    affine = [p for pair in model.loop_norm_layers() for layer in pair for p in (layer.weight, layer.bias)]
    if (len(model.blocks) != 1 or len(affine) != 16 or len({id(p) for p in affine}) != 16
            or start["parameters"] != 1598918
            or any(not torch.equal(p, torch.ones_like(p) if i % 2 == 0 else torch.zeros_like(p))
                   for i, p in enumerate(affine))):
        raise ValueError("loop-specific affine identity, count or initialization differs")
    named = dict(model.named_parameters())
    groups = optimizer_for(model, config).param_groups
    optimized = [p for g in groups for p in g["params"]]
    ema = ParameterEMA(model, config["ema_decay"])
    if len(optimized) != len(named) or {id(p) for p in optimized} != {id(p) for p in named.values()} or (
        ema.shadow.keys() != named.keys() or any(p.dtype != torch.float32 for p in ema.shadow.values())
    ):
        raise ValueError("optimizer or EMA parameter coverage differs")
    evidence["base_started"] = reference
    evidence["planned_starts"][CASE] = start | {"config_sha256": json_digest(config)}
    evidence["norm_tensor_sha256"] = state_sha(model, prefixes="loop_norms.")
    evidence["norm_initialization"] = {"affine_tensors_all": len(affine), "added_parameters": 2304,
                                        "optimizer_unique": len(optimized), "ema_fp32_shadows": len(ema.shadow)}
    evidence["matched_native_epochs"] = len(planned)
    tile512, tile768 = torch.zeros(1, 3, 512, 512), torch.zeros(1, 3, 768, 768)
    paths = (("train768_two_reads", tile768, {"depth_midpoint": True}),
             ("val512_four_reads", tile512, {"depth_staged": True}), ("deploy512_final", tile512, {}))
    for label, image, kwargs in paths:
        base_cost = forward_costs(base, image, **kwargs)
        new_cost = forward_costs(model, image, **kwargs)
        if base_cost != new_cost:
            raise ValueError(f"loop norm changed nominal dense FLOPs at {label}")
        evidence.setdefault("dense_paths", {})[label] = new_cost
    phases = ("training", "validation_live", "validation_ema", "initialization", "checkpoint")
    measured = {label: {phase: sum(r["seconds"] for r in e["phases"] if r["phase"] == phase) for phase in phases}
                for label, e in ((SHARED_ARM, s_evidence), (UNSHARED_ARM, u_evidence))}
    projected = {phase: max(x[phase] for x in measured.values()) for phase in phases}
    base_time = sum(projected.values())
    evidence["projection"] = {
        "measured_phases": measured, "projected_phases": projected, "base_seconds": base_time,
        "allowance_seconds": .25 * base_time + 120, "budget_estimate_seconds": 1.25 * base_time + 120,
        "headroom_bytes_estimate": 2 * max(x["execution"]["peak_reserved_bytes"] for x in (s_evidence, u_evidence)),
        "actual_gpu": "not_run",
    }
    evidence["reference_checkpoints"] = {label: e["case"]["retained_checkpoint"]
                                         for label, e in ((SHARED_ARM, s_evidence), (UNSHARED_ARM, u_evidence))}
    return evidence
