import random

import numpy as np
import pytest
import torch

from loopcrowd import gradient_diagnosis as probe
from loopcrowd.manifest import Record
from loopcrowd.model import LoopCrowd
from loopcrowd.probe_budget import ProbeBudget
from loopcrowd.targets import point_targets
from scripts.archive_lc007c import correction
from scripts.prepare_lc007c import crop_plan


def fixture(variant):
    torch.manual_seed(29)
    model = LoopCrowd(variant).eval()
    images = torch.randn((2, 3, 32, 32))
    labels = point_targets([[5, 6], [18, 13]], 32, 32)
    return model, images, {name: torch.stack((value, value)) for name, value in labels.items()}


@pytest.mark.parametrize("variant", ("shared_2", "unshared_2"))
def test_expanded_graph_outputs_native_gradients_and_no_path(variant):
    model, images, labels = fixture(variant)
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    buffers = dict(model.named_buffers())
    rng = torch.get_rng_state()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer_state = optimizer.state_dict()
    evidence = probe.diagnose_two_calls(model, images, labels)
    assert max(evidence["outputs_max_abs"].values()) == 0
    assert max(evidence["losses_max_abs"].values()) == 0
    assert len(evidence["gradient_equivalence"]) == 2
    rows = {(row["comparison"], row["group"]): row for row in evidence["comparisons"]}
    assert rows["final_first_vs_second_call", "decoder"]["status"] == "no_gradient_path"
    assert rows["total_first_vs_second_call", "decoder"]["status"] == "measured"
    assert rows["final_first_vs_second_call", "block"]["status"] == "measured"
    if variant == "unshared_2":
        assert rows["early_vs_final_supervision", "block_2"]["status"] == "no_gradient_path"
        assert rows["final_first_vs_second_call", "block"]["relation"] == "descriptive_across_independent_blocks"
    assert dict(model.named_buffers()).keys() == buffers.keys()
    assert all(not module.training for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert optimizer.state_dict() == optimizer_state
    torch.testing.assert_close(model.state_dict(), saved, atol=0, rtol=0)
    torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)


def test_bf16_ema_parameter_override_keeps_live_parameters():
    model, images, labels = fixture("shared_2")
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    shadow = {name: value.detach().clone() for name, value in model.named_parameters()}
    shadow["point_logits.bias"].add_(.125)
    evidence = probe.diagnose_two_calls(model, images, labels, precision="bf16", parameter_values=shadow)
    assert max(evidence["outputs_max_abs"].values()) == 0
    assert evidence["gradient_equivalence"]["final"]["block"]["max_abs_delta"] >= 0
    assert not torch.equal(shadow["point_logits.bias"], model.point_logits.bias)
    torch.testing.assert_close(model.state_dict(), saved, atol=0, rtol=0)


def test_exception_restores_buffers_modes_and_torch_python_numpy_rng(monkeypatch):
    model, images, labels = fixture("shared_2")
    model.p2[1].train()
    modes = [m.training for m in model.modules()]
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()

    def fail(_model, _parameters, buffers, _images):
        buffers["p2.1.running_mean"].fill_(9)
        random.random()
        torch.rand(2)
        np.random.rand()
        raise RuntimeError("fixture exception")

    monkeypatch.setattr(probe, "_expanded_forward", fail)
    with pytest.raises(RuntimeError, match="fixture exception"):
        probe.diagnose_two_calls(model, images, labels)
    assert [m.training for m in model.modules()] == modes
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, atol=0, rtol=0)
    torch.testing.assert_close(model.state_dict(), saved, atol=0, rtol=0)


def test_gradient_pair_reports_projection_zero_nonfinite_and_disconnected():
    weights = {"x": torch.empty(2)}
    left = {"x": torch.tensor([2., 0.])}
    right = {"x": torch.tensor([-1., 0.])}
    result = probe.gradient_pair(left, right, weights, left_scale=.25)
    assert result["cosine"] == -1 and result["conflict"] is True
    assert result["weighted_left_projection_on_right"] == -.5
    assert result["weighted_left_right_norm_ratio"] == .5
    for x, status in ((torch.zeros(2), "zero_or_near_zero_norm"),
                      (torch.tensor([1e-15, 0.]), "zero_or_near_zero_norm"),
                      (torch.tensor([float("inf"), 0.]), "nonfinite"), (None, "no_gradient_path")):
        result = probe.gradient_pair(left, {"x": x}, weights)
        assert result["status"] == status and result["cosine"] is None and result["conflict"] is None


def test_count_error_cancellation_is_additive_across_signed_crossing():
    before = {"local_fp": 3, "local_fn": 7, "count_error": -4, "abs_count_error": 4}
    after = {"local_fp": 6, "local_fn": 5, "count_error": 1, "abs_count_error": 1}
    result = correction(before, after)
    assert result["correction_abs_error"] == -3
    assert result["delta_fp"] == 3 and result["delta_fn"] == -2 and result["delta_canceled_unmatched"] == 4


def test_train_only_crop_manifest_determinism_and_group_bounds():
    train = [Record(str(i), "synthetic", "train", "unused", 512, 512, [[x + 1, 4] for x in range(i)])
             for i in range(30)]
    plan = crop_plan(train, 42)
    assert crop_plan(list(reversed(train)), 42) == plan
    assert len(plan["batches"]) == 12
    assert [batch["density_group"] for batch in plan["batches"]] == ["low"] * 4 + ["mid"] * 4 + ["high"] * 4
    crops = [crop for batch in plan["batches"] for crop in batch["crops"]]
    assert len({crop["image_id"] for crop in crops}) == 24
    assert all(crop["left"] == crop["top"] == 0 for crop in crops)
    with pytest.raises(ValueError, match="original train"):
        crop_plan(train + [Record("val", "synthetic", "val", "unused", 512, 512, [])], 42)
    with pytest.raises(ValueError, match="eligible"):
        crop_plan(train[:18], 42)


def test_cooperative_budget_accounts_interrupted_phase_and_raises_final_sync():
    times = iter((10.0, 12.5))
    sync_calls = iter((None, RuntimeError("final synchronization fixture")))

    def synchronize():
        error = next(sync_calls)
        if error:
            raise error

    guard = ProbeBudget(seconds=20, reserved_bytes=120, cases=1, synchronize=synchronize,
                        peak_reserved=lambda: 100, clock=lambda: next(times))
    with pytest.raises(RuntimeError, match="final synchronization fixture"):
        with guard.phase("fixture", estimate_seconds=5, batch_policy_case=True):
            raise ValueError("original phase failure")
    assert guard.seconds == 2.5 and guard.cases == 1
    assert guard.phases[0]["status"] == "synchronization_or_allocator_check_failed"
    with pytest.raises(RuntimeError, match="would be exceeded"):
        with guard.phase("over budget", estimate_seconds=18):
            pytest.fail("must refuse before running")
    with pytest.raises(RuntimeError, match="case cap"):
        with guard.phase("too many cases", estimate_seconds=1, batch_policy_case=True):
            pytest.fail("must refuse before running")
