import random

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.ema import ParameterEMA, preserve_training_context
from loopcrowd.experiment import DEFAULT_CONFIG, load_checkpoint
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.training import fit


def test_fp32_shadow_updates_shared_parameter_once_and_skips_unsuccessful_step():
    block = torch.nn.Linear(2, 2, bias=False).double()
    model = torch.nn.Sequential(block, block)
    block.weight.data.fill_(2)
    ema = ParameterEMA(model, .99)
    assert list(ema.shadow) == ["0.weight"]
    assert ema.shadow["0.weight"].dtype == torch.float32
    block.weight.data.fill_(12)
    ema.update(updated=False)
    torch.testing.assert_close(ema.shadow["0.weight"], torch.full((2, 2), 2.0))
    assert ema.step == 0
    ema.update(updated=True)
    torch.testing.assert_close(ema.shadow["0.weight"], torch.full((2, 2), 2.10))
    assert ema.step == 1
    ema.shadow["0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        ema.require_finite()


@pytest.mark.parametrize("decay", (True, 0, 1, float("inf"), float("nan")))
def test_invalid_decay_rejected(decay):
    with pytest.raises(ValueError, match="decay"):
        ParameterEMA(torch.nn.Linear(2, 2), decay)


def test_ema_checkpoint_requires_all_finite_parameters_and_exact_update_position():
    model = torch.nn.Linear(2, 2)
    ema = ParameterEMA(model, .99)
    ema.update(updated=True)
    state = ema.state_dict()
    other = ParameterEMA(model, .99)
    for bad in (state | {"decay": .9}, state | {"step": 0}, state | {"shadow": {"weight": state["shadow"]["weight"]}}):
        with pytest.raises(ValueError, match="EMA checkpoint"):
            other.load_state_dict(bad, expected_step=1)
        assert other.step == 0
    other.load_state_dict(state, expected_step=1)
    torch.testing.assert_close(other.state_dict(), state, rtol=0, atol=0)


def test_exception_restores_live_parameters_modes_buffers_and_rng():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)).train()
    model[1].eval()
    ema = ParameterEMA(model, .99)
    model[0].weight.data.add_(7)
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    state = torch.get_rng_state()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError, match="fixture"):
        with preserve_training_context(model, torch.device("cpu")), ema.use_shadow():
            model.eval()
            torch.testing.assert_close(model[0].weight, ema.shadow["0.weight"], rtol=0, atol=0)
            torch.testing.assert_close(model[1].running_var, initial["1.running_var"], rtol=0, atol=0)
            torch.rand(1)
            np.random.rand()
            random.random()
            raise RuntimeError("fixture")
    assert model.training and model[0].training and not model[1].training
    torch.testing.assert_close(model.state_dict(), initial, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


def test_training_entrypoint_preserves_live_state_and_exact_ema_resume(tmp_path):
    image = tmp_path / "fixture.png"
    Image.new("RGB", (33, 37), (170, 17, 75)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    records = [Record(str(i), "fixture", split, image.name, 33, 37,
                      [[10., 12.], [24., 20.]][:n])
               for i, (split, n) in enumerate((("train", 1), ("train", 2), ("val", 2)))]
    write_manifest(manifest, records)
    config = DEFAULT_CONFIG | {"variant": "shared_2", "early_loss_weight": .25, "epochs": 3,
                               "tile_size": 32, "tile_overlap": 8, "fixed_training_horizon": True}
    enabled = config | {"ema_decay": .99}
    live, paired, paused = (tmp_path / name for name in ("live", "paired", "paused"))
    device = torch.device("cpu")
    fit(config, manifest, live, device)
    fit(enabled, manifest, paired, device)
    fit(enabled, manifest, paused, device, stop_after=1)
    fit(enabled, manifest, paused, device, resume=True)
    assert not (paired / "best.pt").exists()
    assert len(list((paired / "images").glob("*.json"))) == 3
    states = [load_checkpoint(output / "last.pt", device) for output in (live, paired, paused)]
    assert "ema" not in states[0] and states[1]["ema"]["step"] == 3
    for key in ("model", "optimizer", "scaler"):
        for other in states[1:]:
            torch.testing.assert_close(states[0][key], other[key], rtol=0, atol=0)
    for key in ("scheduler", "global_step", "initial_state_sha256"):
        assert states[0][key] == states[1][key] == states[2][key]
    torch.testing.assert_close(states[1]["ema"], states[2]["ema"], rtol=0, atol=0)
    for plain, on, resumed in zip(*(state["history"] for state in states)):
        for key in ("crop_order_sha256", "learning_rates", "loss", "gradient_l2", "optimizer_updates"):
            assert plain[key] == on[key] == resumed[key]
        for metric in ("count", "nap_local_v1", "local_tp", "local_fp", "local_fn"):
            assert plain["validation"][metric] == on["validation"][metric] == resumed["validation"][metric]
        assert on["ema_validation"] == resumed["ema_validation"]
