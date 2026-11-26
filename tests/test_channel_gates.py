import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.calibration import json_digest
from loopcrowd.ema import ParameterEMA
from loopcrowd.experiment import DEFAULT_CONFIG, load_checkpoint, make_model, read_config, seed_all, state_sha
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.model import CHANNEL_GATES, LoopCrowd
from loopcrowd.targets import point_loss, point_targets
from loopcrowd.training import fit, supervised_point_loss
from scripts.prepare_lc007d import CONFIG, initialization
from scripts.profile import forward_costs
from scripts.run_lc007d import ScreenBudget, require_authorization


SCREEN = json.loads(CONFIG.read_text(encoding="utf-8"))
VARIANTS = [(spec["variant"], "unshared_2" if arm == "U-RS" else "shared_2", spec["gate_parameters"])
            for arm, spec in SCREEN["variants"].items()]


@pytest.mark.parametrize("variant,reference,extra", VARIANTS)
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_identity_gates_preserve_initialization_outputs_loss_and_gradients(variant, reference, extra, precision):
    seed_all(42)
    legacy = LoopCrowd(reference)
    legacy_init = initialization(legacy)
    seed_all(42)
    gated = LoopCrowd(variant)
    gated_init = initialization(gated)
    for name in ("common_sha256", "encoder_sha256", "decoder_sha256", "block_sha256", "rng_after_init_sha256"):
        assert legacy_init[name] == gated_init[name]
    if reference == "unshared_2":
        assert len(set(gated_init["block_sha256"])) == 2
    deltas = gated.channel_gate_deltas
    assert deltas.dtype == torch.float32 and torch.count_nonzero(deltas) == 0
    assert deltas.numel() == extra == gated_init["parameters"] - legacy_init["parameters"]
    assert set(gated.state_dict()) - set(legacy.state_dict()) == {"channel_gate_deltas"}

    image = torch.randn(2, 3, 32, 32)
    target = {key: value.repeat((2,) + (1,) * value.ndim)
              for key, value in point_targets(np.array([[8.25, 13.75], [22., 25.]]), 32, 32, 2).items()}
    stages, losses, dtypes = [], [], []
    cfg = DEFAULT_CONFIG | {"early_loss_weight": 0.25}
    for model in (legacy, gated):
        seen = []
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
            outputs = model(image, return_stages=True, trace=lambda *trace: seen.append(tuple(x.dtype for x in trace[1:])))
            early = point_loss(outputs["early"], target)
            final = point_loss(outputs["final"], target)
            total = supervised_point_loss(model, image, target, cfg)
        if model is gated:
            egrad, = torch.autograd.grad(early, deltas, retain_graph=True)
            assert torch.isfinite(egrad).all()
            assert torch.all(egrad[0].norm(dim=-1) > 0)
            assert torch.count_nonzero(egrad[1]) == 0
        total.backward()
        stages.append(outputs)
        losses.append(torch.stack((early, final, total)))
        dtypes.append(seen)

    tolerances = SCREEN["cpu_equivalence_tolerances"][precision]
    for stage in ("early", "final"):
        for field in ("logits", "offsets"):
            torch.testing.assert_close(stages[0][stage][field], stages[1][stage][field], **tolerances["output"])
    torch.testing.assert_close(*losses, **tolerances["output"])
    assert dtypes[0] == dtypes[1]
    if precision == "bf16":
        assert all(all(dtype == torch.bfloat16 for dtype in stage) for stage in dtypes[1])
    gated_parameters = dict(gated.named_parameters())
    for name, parameter in legacy.named_parameters():
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, gated_parameters[name].grad, **tolerances["gradient"])
    assert torch.isfinite(deltas.grad).all() and torch.all(deltas.grad.norm(dim=-1) > 0)


@pytest.mark.parametrize("variant,reference,extra", VARIANTS)
def test_nonuniform_scales_and_post_state_readout(variant, reference, extra):
    model = LoopCrowd(variant).eval()
    image = torch.randn(1, 3, 32, 40)
    with pytest.raises(ValueError, match="both fixed loop calls"):
        model(image, loop_count=1)
    deltas = model.channel_gate_deltas
    branches = CHANNEL_GATES[variant]
    scales = torch.linspace(-0.35, 0.25, deltas.numel()).reshape_as(deltas)
    with torch.no_grad():
        deltas.copy_(scales)
        before, after, gates = [], [], []

        def trace(index, previous, current, names):
            assert index == len(before)
            before.append(previous)
            after.append(current)
            gates.append(names)

        stages = model(image, return_stages=True, channel_trace=trace)
        p2 = model.p2(image)
        tokens = model.project(model.p3(p2)).permute(0, 2, 3, 1)
        for index, stage in enumerate(("early", "final")):
            block = model.blocks[0 if model.shared else index]
            blocks = []
            block(tokens, trace=lambda x, y, a, b: blocks.append(a))
            first = tokens + blocks[0] * (1 + deltas[index, 0])
            residual = first + block.ffn(block.norm2(first)) * (1 + deltas[index, 1])
            post = residual * (1 + deltas[index, 2]) if "state" in branches else residual
            torch.testing.assert_close(before[index], tokens)
            torch.testing.assert_close(after[index], post)
            assert tuple(gates[index]) == branches
            for branch, scale in zip(branches, 1 + deltas[index], strict=True):
                torch.testing.assert_close(gates[index][branch], scale)
            for field, result in model._decode(p2, post).items():
                torch.testing.assert_close(stages[stage][field], result)
            tokens = post
    profile = forward_costs(model, image)
    assert profile["dense_flops"] == forward_costs(LoopCrowd(reference).eval(), image)["dense_flops"]
    assert profile["gate_adds"] == extra and profile["gate_multiplies"] == extra * 4 * 5


@pytest.mark.parametrize("variant,reference,extra", VARIANTS)
def test_new_config_and_unique_ema_checkpoint(tmp_path, variant, reference, extra):
    path = Path(f"configs/lc006_{reference}_aux_025.json")
    config = read_config(path, variant) | {"pretrained": False, "ema_decay": 0.99}
    model = make_model(config, initialize=False)
    ema = ParameterEMA(model, config["ema_decay"])
    assert len(ema.shadow) == len({id(parameter) for parameter in model.parameters()})
    assert ema.shadow["channel_gate_deltas"].numel() == extra
    with torch.no_grad():
        model.channel_gate_deltas.add_(0.5)
    ema.update(updated=True)
    saved = tmp_path / "model.pt"
    torch.save({"config": config, "model": model.state_dict(), "ema": ema.state_dict()}, saved)
    state = load_checkpoint(saved, torch.device("cpu"))
    restored = make_model(state["config"], initialize=False)
    restored.load_state_dict(state["model"])
    shadow = ParameterEMA(restored, config["ema_decay"])
    shadow.load_state_dict(state["ema"], expected_step=1)
    before = state_sha(restored)
    with shadow.use_shadow():
        torch.testing.assert_close(restored.channel_gate_deltas, torch.full_like(model.channel_gate_deltas, 0.005))
    assert state_sha(restored) == state_sha(model) == before


def test_synthetic_ema_fit_records_post_gate_statistics(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    image = tmp_path / "fixture.png"
    Image.new("RGB", (32, 32), (90, 80, 20)).save(image)
    write_manifest(manifest, [Record(str(i), "synthetic", split, image.name, 32, 32, [[10., 12.]])
                              for i, split in enumerate(("train", "train", "val"))])
    config = DEFAULT_CONFIG | {"variant": "shared_2_channel_residual_state", "early_loss_weight": 0.25,
                               "ema_decay": 0.99, "audit_training": True, "fixed_training_horizon": True,
                               "tile_size": 32, "tile_overlap": 8, "batch_size": 2}
    output = tmp_path / "fit"
    seed_all(config["seed"])
    initial = state_sha(make_model(config, initialize=False))
    expected = {"initial_sha256": initial, "trajectory": [("incorrect", "incorrect")]}
    with pytest.raises(ValueError, match="observed crop order or LR differs"):
        fit(config, manifest, tmp_path / "mismatch", torch.device("cpu"), expected=expected)
    order = torch.utils.data.DataLoader(range(2), batch_size=2, shuffle=True,
                                       generator=torch.Generator().manual_seed(config["seed"] + 1))
    ids = [(index, 0, 0) for index in next(iter(order)).tolist()]
    digest = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    expected["trajectory"] = [(digest, json_digest([config["lr"]]))]
    run = fit(config, manifest, output, torch.device("cpu"), expected=expected)
    assert run["global_step"] == 1
    row, = json.loads((output / "history.json").read_text(encoding="utf-8"))
    stats = row["channel_gate_statistics"]
    assert set(stats) == {"live", "ema"}
    for policy in stats.values():
        assert [(row["stage"], row["branch"]) for row in policy] == [
            (stage, branch) for stage in (1, 2) for branch in CHANNEL_GATES[config["variant"]]]
        assert all(row["min"] < row["max"] and row["delta_l2"] > 0 for row in policy)
    assert stats["ema"][0]["delta_l2"] < stats["live"][0]["delta_l2"]


def test_screen_requires_separate_matching_authorization_and_charges_failed_phase(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="separate owner authorization"):
        require_authorization({}, SCREEN, "config", "commit")
    record = {"owner_message_ref": "test fixture", "approved_at": "test fixture", "approval_scope": "test fixture",
              "issue": 23, "code_commit": "commit", "config_sha256": "config"}
    with pytest.raises(ValueError, match="does not match"):
        require_authorization(record, SCREEN, "config", "commit")
    ledger = tmp_path / "phases.jsonl"
    budget = ScreenBudget(SCREEN["limits_if_approved"], ledger, torch.device("cpu"))
    assert not budget.active
    with pytest.raises(RuntimeError, match="fixture"):
        with budget.phase("preflight"):
            raise RuntimeError("fixture")
    assert json.loads(ledger.read_text(encoding="utf-8"))["failure"] == "RuntimeError: fixture"
    monkeypatch.setattr(budget, "updates", SCREEN["limits_if_approved"]["total_updates"])
    with pytest.raises(RuntimeError, match="optimizer-update cap"):
        budget.check("train_batch")
