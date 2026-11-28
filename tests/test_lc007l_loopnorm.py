from copy import deepcopy
from pathlib import Path

import torch
import pytest
from PIL import Image

from loopcrowd.ema import ParameterEMA
from loopcrowd.experiment import load_checkpoint, make_model, read_config, seed_all
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.model import WindowBlock
from loopcrowd.training import fit, optimizer_for
from scripts.archive_lc007b import METRICS
from scripts.archive_lc007l import diagnostic_rows, score
from scripts.prepare_lc007i import recipe as base_recipe
from scripts.prepare_lc007l import SCREEN, recipe
from scripts.run_lc007d import read_json
from scripts.run_lc007l import authorize

PLAN = read_json(SCREEN)
DEVICE = torch.device("cpu")


def models():
    seed_all(42)
    base = make_model(base_recipe(PLAN), initialize=False).eval()
    before = torch.get_rng_state()
    seed_all(42)
    new = make_model(recipe(PLAN), initialize=False).eval()
    assert torch.equal(before, torch.get_rng_state())
    return base, new


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_initial_states_and_readouts(dtype):
    base, new = models()
    cpu = torch.randn(1, 3, 49, 57)
    expected, actual = [], []
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        x = base.forward_depth_stages(cpu, diagnostic=True, state_trace=lambda _, a, b: expected.append(b))
        y = new.forward_depth_stages(cpu, diagnostic=True, state_trace=lambda _, a, b: actual.append(b))
        two = new.forward_depth_stages(cpu)
        one = new(cpu)
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(two, {k: y[k] for k in ("early", "final")}, rtol=0, atol=0)
    torch.testing.assert_close(one, y["final"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="all four"):
        new(cpu, loop_count=2)


def test_norm_override_preserves_fixed_and_shifted_irregular_block():
    block = WindowBlock().eval()
    x = torch.randn(2, 9, 13, 192)
    norms = (deepcopy(block.norm1), deepcopy(block.norm2))
    with torch.no_grad():
        for shift in (0, 4):
            torch.testing.assert_close(block(x, shift), block(x, shift, norms=norms), atol=0, rtol=0)
    assert all(n.eps == o.eps and n.normalized_shape == o.normalized_shape == (192,)
               for n, o in zip(norms, (block.norm1, block.norm2), strict=True))


def test_all_affines_active_and_shared_gradients_aggregate_exactly():
    base, new = models()
    assert sum(p.numel() for p in new.parameters()) == 1598918
    assert sum(p.numel() for p in base.parameters()) == 1596614
    assert len(new.blocks) == 1 and not base.loop_norm_layers()
    for key, value in base.state_dict().items():
        assert torch.equal(new.state_dict()[key], value)
    norm_params = [p for pair in new.loop_norm_layers() for norm in pair for p in (norm.weight, norm.bias)]
    assert len(norm_params) == len({id(p) for p in norm_params}) == 16
    x = torch.randn(1, 3, 35, 41)
    for model in (base, new):
        y = model.forward_depth_stages(x)
        loss = y["final"]["logits"].mean() + .25 * y["early"]["offsets"].mean()
        loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.norm() > 0 for p in norm_params)
    for field in ("attn", "ffn"):
        for a, b in zip(getattr(base.blocks[0], field).parameters(), getattr(new.blocks[0], field).parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    for branch, old in enumerate((base.blocks[0].norm1, base.blocks[0].norm2)):
        for label in ("weight", "bias"):
            contributions = [getattr(pair[branch], label).grad for pair in new.loop_norm_layers()]
            torch.testing.assert_close(getattr(old, label).grad, sum(contributions), atol=1e-6, rtol=1e-5)
            assert not torch.equal(contributions[0], contributions[-1])
    optimizer_for(new, recipe(PLAN)).step()
    assert any(row[2] > 0 or row[3] > 0 for pair in new.loop_norm_statistics()[1:] for row in pair)


def test_checkpoint_roundtrip_ema_live_bn_and_epoch_diagnostics(tmp_path):
    manifest = tmp_path / "fixture.jsonl"
    Image.new("RGB", (32, 32), (120, 75, 20)).save(tmp_path / "tile.png")
    write_manifest(manifest, [Record(str(i), "fixture", split, "tile.png", 32, 32, [[8.5, 12.5]])
                              for i, split in enumerate(("train", "train", "val"))])
    config = recipe(PLAN) | {"pretrained": False, "precision": "fp32", "train_crop_size": 32, "tile_size": 32,
                             "tile_overlap": 8, "epochs": 2, "workers": 0, "lr_schedule": "constant"}
    root = tmp_path / "fit"
    fit(config, manifest, root, DEVICE, stop_after=1)
    checkpoint = load_checkpoint(root / "last.pt", DEVICE)
    model = make_model(config, initialize=False).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    ema = ParameterEMA(model, .99)
    ema.load_state_dict(checkpoint["ema"], expected_step=1)
    assert ema.shadow.keys() == dict(model.named_parameters()).keys()
    assert all(x.dtype == torch.float32 for x in ema.shadow.values())
    buffers = deepcopy(dict(model.named_buffers()))
    with ema.use_shadow():
        assert all(torch.equal(buffers[k], x) for k, x in model.named_buffers())
    rows = list(diagnostic_rows(read_json(root / "history.json")))
    assert len(rows) == 16 and any(r["gamma_vs_loop1_l2"] > 0 for r in rows)
    legacy = {k: v for k, v in config.items() if k != "loop_norm"}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        make_model(legacy, initialize=False).load_state_dict(checkpoint["model"], strict=True)
    with pytest.raises(ValueError, match="resume requires matching"):
        fit(legacy, manifest, root, DEVICE, resume=True)


def test_invalid_config_and_previous_scope_stay_rejected(tmp_path):
    config = recipe(PLAN)
    from loopcrowd.experiment import write_json
    path = tmp_path / "config.json"
    write_json(path, config)
    assert read_config(path)["loop_norm"] and read_config(path)["train_crop_size"] == 768
    with pytest.raises(ValueError, match="ungated shifted shared"):
        make_model(config | {"variant": "shifted_unshared_4"}, initialize=False)
    with pytest.raises(ValueError, match="ungated shifted shared"):
        make_model(config | {"refinement": "guided"}, initialize=False)
    old = read_json(Path("reports/LC-007K/training_results/authorization.json"))
    with pytest.raises(ValueError):
        authorize(old, PLAN, "sha", "head", "tree")


def test_raw_win_remains_separate_from_practical_and_full_gate():
    common = dict.fromkeys(METRICS, 100) | {"policy": "ema", "stage": "final", "density_group": "all"}
    def rows(mae, nap):
        return [common | {"epoch": e, "mae": mae, "local_nap": nap} for e in range(36, 41)]
    training = [{"epoch": e, "final_loss": 1.} for e in range(31, 41)]
    s, u = {"metrics": rows(19.7, .346196)}, {"metrics": rows(19.16, .351294)}
    outcome = score(training, rows(19., .36), PLAN["decision"], s768=s, u768=u)
    assert all(outcome["comparisons"]["U4-shift-768"]["raw"].values())
    assert not outcome["comparisons"]["U4-shift-768"]["practical"]["mae"] and outcome["decision"] == "fail"
    assert score(training, rows(18.58, .36), PLAN["decision"], s768=s, u768=u)["decision"] == "pass"
