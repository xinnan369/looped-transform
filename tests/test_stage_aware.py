import numpy as np
import pytest
import torch

from loopcrowd.diagnostics import LoopStatistics, _match_counts
from loopcrowd.experiment import DEFAULT_CONFIG, make_model, read_config
from loopcrowd.model import LoopCrowd


def test_frozen_backbone_bn_buffers_and_affine_gradients(tmp_path):
    images = torch.randn(2, 3, 32, 40)
    config = tmp_path / "old.json"
    config.write_text("{}")
    assert read_config(config)["backbone_bn"] == "train"
    ordinary = make_model({key: val for key, val in DEFAULT_CONFIG.items() if key != "backbone_bn"}, True)
    frozen = LoopCrowd("cnn_only", freeze_backbone_bn=True)
    frozen.load_state_dict(ordinary.state_dict())
    frozen.eval().train()
    norms = [layer for layer in frozen.modules() if isinstance(layer, torch.nn.BatchNorm2d)]
    assert frozen.training and norms and all(not layer.training for layer in norms)
    buffers = {key: value.clone() for key, value in frozen.named_buffers()}
    frozen(images)["logits"].square().mean().backward()
    assert all(torch.equal(value, buffers[key]) for key, value in frozen.named_buffers())
    assert all(layer.weight.grad is not None and layer.bias.grad is not None for layer in norms)
    assert any(layer.weight.grad.abs().sum() > 0 and layer.bias.grad.abs().sum() > 0 for layer in norms)
    ordinary.train()(images)
    assert any(not torch.equal(value, buffers[key]) for key, value in ordinary.named_buffers())


@pytest.mark.parametrize("variant,stages", [("shared_2_tied_gain", 1), ("shared_2_stage_gain", 2)])
def test_unit_gains_match_original_and_both_calls_share_core(variant, stages):
    torch.manual_seed(42)
    old = LoopCrowd("shared_2")
    torch.manual_seed(42)
    gained = LoopCrowd(variant)
    assert gained.gain_logits.shape == (stages, 2)
    assert all(torch.equal(value, gained.state_dict()[key]) for key, value in old.state_dict().items())
    images = torch.randn(2, 3, 40, 40)
    calls = []
    gained.blocks[0].register_forward_pre_hook(lambda module, args: calls.append(module.attn.in_proj_weight))
    ordinary, experimental = old(images), gained(images)
    for field in ordinary:
        torch.testing.assert_close(ordinary[field], experimental[field], atol=0, rtol=0)
    for outputs in (ordinary, experimental):
        (outputs["logits"].square().mean() + outputs["offsets"].mean()).backward()
    for name, param in old.named_parameters():
        torch.testing.assert_close(param.grad, dict(gained.named_parameters())[name].grad, atol=0, rtol=0)
    assert len(calls) == 2 and calls[0] is calls[1]
    assert gained.gain_logits.grad.abs().min() > 0


def test_stage_indexing_and_fixed_prefixes():
    model = LoopCrowd("shared_2_stage_gain").eval()
    with torch.no_grad():
        model.gain_logits[1] = torch.tensor([1.0, -1.0])
        assert torch.equal(model.loop_gains()[0], torch.ones(2))
        assert model.gain_logits[0].data_ptr() != model.gain_logits[1].data_ptr()
        inputs = torch.randn(1, 3, 40, 40)
        trace = LoopStatistics()
        full = model(inputs, loop_count=2, trace=trace)
        one = model(inputs, loop_count=1)
        zero = model(inputs, loop_count=0)
        ordinary = model(inputs)
        assert trace.result().keys() == {"1", "2"}
        assert torch.equal(full["logits"], ordinary["logits"])
        assert not torch.equal(one["logits"], zero["logits"])
        assert not torch.equal(full["logits"], one["logits"])
        model.gain_logits[1].fill_(-100)
        assert torch.equal(model(inputs)["logits"], one["logits"])
    with pytest.raises(ValueError, match="fixed depth"):
        model(inputs, loop_count=3)


def test_competing_predictions_keep_distinct_close_matches():
    points = np.array([[0., 0.], [1.8, 0.], [1.8, 8.]])
    candidates = np.array([[0., 0., .9], [1.8, 0., .8], [0.1, 0., .7], [80., 80., .6]])
    result = _match_counts(points, candidates)
    assert (result["local_tp"], result["local_fp"], result["local_fn"]) == (2, 2, 1)
    assert result["unmatched_predictions_with_eligible_gt"] == 1
    assert result["unmatched_predictions_outside_all_gt_radii"] == 1
    assert result["within_2px_pairs_distinct_matched_gt"] == 1
