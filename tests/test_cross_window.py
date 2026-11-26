import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.ema import ParameterEMA
from loopcrowd.experiment import load_checkpoint, make_model, read_config, seed_all, state_sha
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.model import LoopCrowd, WindowBlock, window_partition, window_reverse
from loopcrowd.targets import point_loss, point_targets
from loopcrowd.training import fit
from scripts.prepare_lc007g import SCREEN
from scripts.probe_cross_window import probe
from scripts.run_lc007d import require_authorization
from scripts.run_lc007e import configure_device_memory
from scripts.run_lc007g import CrossWindowBudget

PLAN = json.loads(SCREEN.read_text(encoding="utf-8"))
CONFIG = Path(PLAN["arm_config"])


@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_token_only_reach_and_gradient_cross_old_window_boundary(precision):
    evidence = probe(precision)
    fixed, shifted = evidence["fixed"], evidence["shifted"]
    assert [entry["possible_tokens"] for entry in fixed] == [64] * 4
    assert [entry["possible_tokens"] for entry in shifted] == [64, 144, 256, 323]
    assert all(entry["max_outside_oracle"] == 0 for entry in fixed + shifted)


@pytest.mark.parametrize("shape", ((1, 1), (3, 11), (17, 19)))
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_shifted_padding_never_feeds_valid_queries(shape, precision):
    torch.manual_seed(42)
    block = WindowBlock()
    tokens = torch.randn(2, *shape, 192)
    if precision == "bf16":
        tokens = tokens.bfloat16()
    windows, mask = window_partition(tokens, 8, 4)
    assert not mask.all(dim=1).any()
    assert torch.equal(window_reverse(windows, tokens.shape, 8, 4), tokens)
    altered = windows.clone()
    altered[mask] = torch.linspace(-10, 10, 192).to(tokens.dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        base, _ = block.attn(windows, windows, windows, key_padding_mask=mask, need_weights=False)
        changed, _ = block.attn(altered, altered, altered, key_padding_mask=mask, need_weights=False)
    assert torch.isfinite(base).all() and torch.isfinite(changed).all()
    torch.testing.assert_close(base[~mask], changed[~mask], atol=0, rtol=0)


@pytest.mark.parametrize("variant", ("shared_4", "unshared_4"))
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_shifted_stage_one_parity_and_midpoint_gradient_isolation(variant, precision):
    seed_all(42)
    fixed = make_model(read_config(CONFIG, variant), initialize=False)
    fixed_rng = torch.get_rng_state()
    seed_all(42)
    shifted_name = f"shifted_{variant}"
    shifted = make_model(read_config(CONFIG, shifted_name), initialize=False)
    assert torch.equal(torch.get_rng_state(), fixed_rng) and state_sha(fixed) == state_sha(shifted)
    assert [state_sha(block) for block in fixed.blocks] == [state_sha(block) for block in shifted.blocks]
    if not shifted.shared:
        assert len({state_sha(block) for block in shifted.blocks}) == 4
    image = torch.randn(2, 3, 72, 80)
    target = {key: torch.stack((value, value)) for key, value in
              point_targets(np.array([[12.25, 18.5], [48., 54.]]), 80, 72, 2).items()}
    data, stage_one_grads = [], []
    for model in (fixed, shifted):
        token_states, calls = [], []
        handles = [block.register_forward_pre_hook(lambda module, args: calls.append(
            (module.attn.in_proj_weight, args[1]))) for block in model.blocks]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
            outputs = model.forward_depth_stages(
                image, diagnostic=True, state_trace=lambda i, before, after: token_states.append(after))
            first = point_loss(outputs["stage1"], target)
            mid = point_loss(outputs["early"], target)
        data.append(outputs)
        stage_one_grads.append(torch.autograd.grad(first, tuple(model.parameters()), allow_unused=True, retain_graph=True))
        mid_path = torch.autograd.grad(mid, tuple(token_states), allow_unused=True)
        assert all(grad is not None and torch.isfinite(grad).all() for grad in mid_path[:2])
        assert mid_path[2:] == (None, None)
        assert [shift for _, shift in calls] == ([0, 4, 0, 4] if model is shifted else [0] * 4)
        assert len({id(param) for param, _ in calls}) == (1 if model.shared else 4)
        assert all(torch.isfinite(state).all() for state in token_states)
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(data[0]["stage1"], data[1]["stage1"], rtol=0, atol=0)
    for before, after in zip(*stage_one_grads, strict=True):
        if before is None:
            assert after is None
        else:
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.equal(data[0]["early"]["logits"], data[1]["early"]["logits"])


def test_shifted_depth_opt_in_and_ema_checkpoint(tmp_path):
    variant = "shifted_shared_4"
    images = torch.zeros(1, 3, 32, 32)
    legacy = LoopCrowd(variant).eval()
    with pytest.raises(ValueError, match="supported fixed-window"):
        legacy.forward_depth_stages(images)
    config = read_config(CONFIG, variant) | {"pretrained": False, "precision": "fp32", "epochs": 1,
                                             "workers": 0, "tile_size": 32, "tile_overlap": 8, "lr_schedule": "constant"}
    opt_in = make_model(config, initialize=False).eval()
    opt_in.load_state_dict(legacy.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(legacy(images), opt_in.forward_depth_stages(images)["final"], atol=0, rtol=0)
    manifest = tmp_path / "fixture.jsonl"
    Image.new("RGB", (32, 32), (100, 80, 25)).save(tmp_path / "tile.png")
    write_manifest(manifest, [Record(str(i), "synthetic", split, "tile.png", 32, 32, [[8.25, 15.5]])
                              for i, split in enumerate(("train", "train", "val"))])
    output, device = tmp_path / "run", torch.device("cpu")
    fit(config, manifest, output, device)
    saved = load_checkpoint(output / "last.pt", device)
    restored = make_model(saved["config"], initialize=False)
    restored.load_state_dict(saved["model"])
    ema = ParameterEMA(restored, .99)
    ema.load_state_dict(saved["ema"], expected_step=1)
    assert len(ema.shadow) == len({id(parameter) for parameter in restored.parameters()})
    assert set(saved["history"][0]["ema_validation"]) == {"stage1", "early", "stage3", "final"}


def test_observational_memory_policy_uses_free_headroom_and_no_fraction(tmp_path, monkeypatch):
    limits = PLAN["limits_if_approved"]
    budget = CrossWindowBudget(limits, tmp_path / "ledger.jsonl", torch.device("cpu"))
    budget.reserved = 3 * 1024**3
    budget.check("validation_image")
    free = 1536 * 1024**2
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda *args: pytest.fail("allocator quota must not be set"))
    assert configure_device_memory(budget.device, None, 1024**3)["free_bytes"] == free
    with pytest.raises(RuntimeError, match="below the projected"):
        configure_device_memory(budget.device, None, free + 1)
    approval = {"issue": 29, "code_commit": "fixture", "config_sha256": "fixture", "approved_at": "fixture",
                "owner_message_ref": "fixture", "approval_scope": "fixture", "limits": limits}
    require_authorization(approval, PLAN, "fixture", "fixture")
    old_limits = limits | {"peak_reserved_bytes": 2 * 1024**3}
    with pytest.raises(ValueError, match="does not match"):
        require_authorization(approval | {"limits": old_limits}, PLAN, "fixture", "fixture")
