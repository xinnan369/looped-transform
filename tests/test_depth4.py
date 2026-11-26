import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.ema import ParameterEMA
from loopcrowd.experiment import DEFAULT_CONFIG, load_checkpoint, read_config, seed_all, state_sha
from loopcrowd.inference import predict_image, predict_image_depth
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.model import LoopCrowd
from loopcrowd.targets import point_loss, point_targets
from loopcrowd.training import fit
from scripts.prepare_lc007e import SCREEN
from scripts.profile import forward_costs
from scripts.run_lc007d import require_authorization
from scripts.run_lc007e import DepthBudget


PLAN = json.loads(SCREEN.read_text(encoding="utf-8"))
CONFIG = Path(PLAN["arm_config"])


@pytest.mark.parametrize("ungated", ("shared_4", "unshared_4"))
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_four_calls_unit_gate_and_gradients(ungated, precision):
    gated = f"{ungated}_channel_residual_state"
    seed_all(42)
    baseline = LoopCrowd(ungated)
    rng = torch.get_rng_state()
    seed_all(42)
    model = LoopCrowd(gated)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(model.blocks) == (1 if model.shared else 4)
    assert len({id(block.attn.in_proj_weight) for block in model.blocks}) == len(model.blocks)
    assert [state_sha(block) for block in model.blocks] == [state_sha(block) for block in baseline.blocks]
    gates = model.channel_gate_deltas
    assert gates.dtype == torch.float32 and gates.numel() == 2304 and torch.count_nonzero(gates) == 0
    images = torch.randn(2, 3, 32, 32)
    target = {k: torch.stack((v, v)) for k, v in point_targets(np.array([[8.25, 13.75]]), 32, 32, 2).items()}
    outputs, losses, call_states = {}, {}, []

    def track(stage, before, after):
        assert stage == len(call_states)
        assert after.dtype == before.dtype
        after.retain_grad()
        call_states.append(after)

    for candidate in (baseline, model):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
            stages = candidate.forward_depth_stages(images, diagnostic=True,
                                                    state_trace=track if candidate is model else None)
            mid, final = (point_loss(stages[key], target) for key in ("early", "final"))
            total = final + .25 * mid
        if candidate is model:
            grad, = torch.autograd.grad(mid, gates, retain_graph=True)
            assert torch.all(grad[:2].norm(dim=-1) > 0) and torch.count_nonzero(grad[2:]) == 0
        total.backward()
        outputs[candidate], losses[candidate] = stages, torch.stack((mid, final, total))
    tol = PLAN["tolerances"][precision]
    for stage in outputs[model]:
        torch.testing.assert_close(outputs[model][stage], outputs[baseline][stage], **tol["output"])
    torch.testing.assert_close(losses[model], losses[baseline], **tol["output"])
    for key, parameter in baseline.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(model.named_parameters())[key].grad, **tol["gradient"])
    assert len(call_states) == 4 and all(state.grad.norm() > 0 for state in call_states)
    assert torch.isfinite(gates.grad).all() and torch.all(gates.grad.norm(dim=-1) > 0)
    if precision == "bf16":
        assert all(state.dtype == torch.bfloat16 for state in call_states)


@pytest.mark.parametrize("arm", list(PLAN["arms"]))
def test_midpoint_and_four_readout_share_one_tiled_pass(arm):
    variant = PLAN["arms"][arm]["variant"]
    model = LoopCrowd(variant).eval()
    config = read_config(CONFIG, variant) | {"tile_size": 32, "tile_overlap": 8, "precision": "fp32"}
    blocks, decoded = [], []
    handles = [block.register_forward_hook(lambda module, args, result: blocks.append(id(module)))
               for block in model.blocks]
    handles.append(model.point_logits.register_forward_hook(lambda *args: decoded.append(1)))
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        two = model.forward_depth_stages(inputs)
        assert set(two) == {"early", "final"} and len(decoded) == 2
        assert len(blocks) == 4 and len(set(blocks)) == len(model.blocks)
        decoded.clear()
        with Image.new("RGB", (31, 29), (150, 33, 84)) as image:
            four, norms = predict_image_depth(model, image, config, torch.device("cpu"))
            assert len(decoded) == 4 and len(blocks) == 8
            plain = predict_image(model, image, config, torch.device("cpu"))
        assert norms.shape == (1, 4, 2) and (norms > 0).all()
        assert tuple(four) == ("stage1", "early", "stage3", "final")
        np.testing.assert_array_equal(four["final"].candidates, plain.candidates)
        assert forward_costs(model, inputs, depth_staged=True)["dense_flops"] > forward_costs(model, inputs)["dense_flops"]
    for handle in handles:
        handle.remove()
    with pytest.raises(ValueError, match="two-call"):
        model.forward_stages(inputs)


def test_depth_ema_fit_and_budget_stop(tmp_path):
    device = torch.device("cpu")
    manifest = tmp_path / "fixture.jsonl"
    Image.new("RGB", (32, 32), (100, 80, 25)).save(tmp_path / "tile.png")
    write_manifest(manifest, [Record(str(i), "synthetic", split, "tile.png", 32, 32, [[8.25, 13.75]])
                              for i, split in enumerate(("train", "train", "val"))])
    variant = "shared_4_channel_residual_state"
    config = DEFAULT_CONFIG | {"variant": variant, "depth_screen": True, "tile_size": 32, "tile_overlap": 8,
                               "early_loss_weight": .25, "ema_decay": .99, "fixed_training_horizon": True}
    with pytest.raises(ValueError, match="owner authorization"):
        require_authorization({}, PLAN, "config", "commit")
    tiny = DepthBudget(PLAN["limits_if_approved"] | {"gpu_seconds": 1}, tmp_path / "denied.jsonl", device)
    with pytest.raises(RuntimeError, match="time budget"):
        tiny.check("train_batch")
    guard = DepthBudget(PLAN["limits_if_approved"], tmp_path / "phases.jsonl", device)
    output = tmp_path / "complete"
    run = fit(config, manifest, output, device, guard=guard)
    assert run["global_step"] == guard.updates == 1
    state = load_checkpoint(output / "last.pt", device)
    model = LoopCrowd(variant)
    model.load_state_dict(state["model"])
    shadow = ParameterEMA(model, .99)
    shadow.load_state_dict(state["ema"], expected_step=1)
    assert len(shadow.shadow) == len({id(p) for p in model.parameters()})
    row, = state["history"]
    for field in ("ema_validation", "depth_live_validation"):
        assert set(row[field]) == {"stage1", "early", "stage3", "final"}
        assert all(stage["state_update"]["nonfinite"] == 0 for stage in row[field].values())
    bad = {"initial_sha256": run["initial_state_sha256"], "trajectory": [("wrong", "wrong")]}
    stopped = tmp_path / "stopped"
    with pytest.raises(ValueError, match="observed crop order or LR"):
        fit(config, manifest, stopped, device, guard=guard, expected=bad)
    assert (stopped / "interrupted.pt").is_file() and (stopped / "mismatched_epoch.json").is_file()
