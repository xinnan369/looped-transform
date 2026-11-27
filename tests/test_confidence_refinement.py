from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.calibration import count_match_metrics
from loopcrowd.ema import ParameterEMA, preserve_training_context
from loopcrowd.experiment import make_model, read_config, seed_all, state_sha
from loopcrowd.identity_diagnostic import confidence_association, count_identity_details, gt_transitions, matched_identities
from loopcrowd.inference import TilePredictions, predict_image, predict_image_depth
from loopcrowd.metrics import NAP_THRESHOLDS, image_matches, point_nap, pooled_point_nap
from loopcrowd.model.refinement import midpoint_uncertainty, refine_update, refinement_gain
from loopcrowd.targets import point_loss, point_targets
from loopcrowd.training import supervised_point_loss

BASE = Path("configs/lc007e_depth4.json")


def config(variant="shifted_shared_4", **changes):
    return read_config(BASE, variant) | changes


def test_pooled_uncertainty_masks_alignment_centering_and_placebo():
    logits = torch.full((2, 2, 3, 5), -5.0, requires_grad=True)
    valid = torch.ones(2, 1, 3, 5, dtype=torch.bool)
    valid[0, :, 2:, :] = False
    valid[0, :, :, 3:] = False
    with torch.no_grad():
        logits[0, 1, 1, 2] = 0
        logits[0, :, 2:, 4] = 0
    score, support = midpoint_uncertainty(logits, valid)
    assert score[0, 0, 1] == 1 and score[0, 1, 2] == 0
    assert score.shape == support.shape == (2, 2, 3)
    gain = refinement_gain(logits, valid)
    placebo = refinement_gain(logits, valid, mode="placebo")
    assert gain.dtype == torch.float32 and not gain.requires_grad
    assert torch.all((gain >= .75) & (gain <= 1.25))
    assert torch.equal(gain[~support], torch.ones_like(gain[~support]))
    assert torch.equal(gain[1], torch.ones_like(gain[1]))
    for sample in range(2):
        assert torch.equal(gain[sample][support[sample]].sort().values, placebo[sample][support[sample]].sort().values)
        torch.testing.assert_close(gain[sample][support[sample]].mean(), torch.tensor(1.0), atol=1e-6, rtol=0)
    assert not torch.equal(gain[0], placebo[0])
    before, after = torch.rand(2, 2, 3, 5).bfloat16(), torch.rand(2, 2, 3, 5).bfloat16()
    assert torch.equal(refine_update(before, after, torch.ones_like(gain)), after)
    assert refine_update(before, after, gain).dtype == torch.bfloat16
    torch.testing.assert_close(refine_update(torch.zeros_like(before), torch.ones_like(after), gain).float(),
                               gain[..., None].expand_as(before), atol=.005, rtol=0)


@pytest.mark.parametrize("precision", ("fp32", "bf16"))
@pytest.mark.parametrize("variant", ("shifted_shared_4", "shifted_unshared_4"))
@pytest.mark.parametrize("mode", ("guided", "placebo"))
def test_refinement_identity_early_parity_and_detached_path(precision, variant, mode):
    seed_all(42)
    base = make_model(config(variant), initialize=False)
    rng = torch.get_rng_state()
    seed_all(42)
    refined = make_model(config(variant, refinement=mode), initialize=False)
    assert state_sha(base) == state_sha(refined) and torch.equal(rng, torch.get_rng_state())
    assert len(refined.blocks) == (1 if refined.shared else 4)
    images = torch.randn(2, 3, 45, 49)
    target = {k: torch.stack((v, v)) for k, v in point_targets(np.array([[8.25, 15.5]]), 49, 45, 2).items()}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        plain = base.forward_depth_stages(images, diagnostic=True)
        states = []
        candidate = refined.forward_depth_stages(images, diagnostic=True, point_valid=target["valid"],
                                                state_trace=lambda _, before, after: states.append(after))
        final = point_loss(candidate["final"], target)
        mid = point_loss(candidate["early"], target)
    for name in ("stage1", "early"):
        torch.testing.assert_close(plain[name], candidate[name], atol=0, rtol=0)
    assert torch.autograd.grad(mid, states[2:], retain_graph=True, allow_unused=True) == (None, None)
    confidence_grad, state_grad = torch.autograd.grad(final, (candidate["early"]["logits"], states[2]),
                                                      allow_unused=True, retain_graph=True)
    assert confidence_grad is None and state_grad.isfinite().all() and state_grad.abs().sum() > 0
    if precision == "bf16":
        assert all(value.dtype == torch.bfloat16 for value in states)


def test_disabled_and_constant_gain_match_legacy_loss_gradient_and_rng():
    model = make_model(config(), initialize=False)
    candidate = deepcopy(model)
    with torch.no_grad():
        for item in (model, candidate):
            item.point_logits.weight.zero_()
            item.point_logits.bias.fill_(-2.)
    candidate.refinement = "guided"
    image = torch.randn(2, 3, 32, 32)
    target = {k: torch.stack((v, v)) for k, v in point_targets(np.array([[8., 15.]]), 32, 32, 2).items()}
    for mode in (None, "guided"):
        replay = deepcopy(candidate)
        replay.refinement = mode
        rng = torch.get_rng_state()
        a, b = supervised_point_loss(deepcopy(model), image, target, config()), supervised_point_loss(
            replay, image, target, config(refinement=mode))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.equal(rng, torch.get_rng_state())
    original = deepcopy(model)
    for item in (original, candidate):
        supervised_point_loss(item, image, target, config(refinement=item.refinement)).backward()
    for a, b in zip(original.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ("guided", "placebo"))
def test_refined_tiled_export_preserves_order_match_ap_and_validation_state(mode):
    cfg = config(refinement=mode, tile_size=32, tile_overlap=8, precision="fp32")
    model = make_model(cfg, initialize=False).eval()
    device = torch.device("cpu")
    with torch.no_grad():
        model.point_logits.weight.zero_()
        model.point_logits.bias.copy_(torch.tensor([.25, -.25]))
    ema = ParameterEMA(model, .99)
    digest, rng = state_sha(model), torch.get_rng_state()
    heads = np.asarray([[16., 15.], [31., 27.]])
    calls = []
    handle = model.point_logits.register_forward_hook(lambda *args: calls.append(1))
    with Image.new("RGB", (39, 29), "white") as frame, preserve_training_context(model, device), ema.use_shadow():
        mids = []
        full, _ = predict_image_depth(model, frame, cfg, device, export_identity=True, midpoint_maps=mids)
        count = len(full["final"].tiles)
        assert len(calls) == 4 * count
        plain, _ = predict_image_depth(model, frame, cfg, device, diagnostic=False)
        assert len(calls) == 6 * count
        observed = predict_image(model, frame, cfg, device)
        assert len(calls) == 8 * count
        for stage in plain:
            a, b = plain[stage], full[stage]
            for key in ("candidates", "tile_ids", "owned"):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            assert a.candidate_ids is None and b.candidate_ids.shape[1] == 4
            rank, gt = image_matches(heads, a.candidates, NAP_THRESHOLDS)
            ids = matched_identities(heads, b)
            np.testing.assert_array_equal(ids["candidate_rows"], rank)
            np.testing.assert_array_equal(ids["gt_ids"], gt)
            expected_ap = point_nap([(heads, a.candidates)])
            assert expected_ap == pooled_point_nap([ids["candidates"][:, 2]], [ids["gt_ids"] >= 0], len(heads))
            active = a.count_candidates()
            expected, = count_match_metrics(heads, active, (.30,))
            counts = count_identity_details(heads, b, .30)
            assert len(counts["matched_ranks"]) == expected["local_tp"]
            assert len(counts["unmatched_gt_ids"]) == expected["local_fn"]
            assert len(counts["false_positive_ranks"]) == expected["local_fp"]
        np.testing.assert_array_equal(observed.candidates, full["final"].candidates)
        assoc = confidence_association(heads, full["early"], full["final"], mids)
        assert np.isfinite(assoc["gt_midpoint_uncertainty"]).all() and len(assoc["gt_id"]) == len(heads)
    handle.remove()
    assert state_sha(model) == digest and torch.equal(torch.get_rng_state(), rng) and not model.training


def test_fp_classes_and_gt_transition_ids_are_separate():
    points = np.array([[4., 4.], [20., 20.], [6., 20.]])
    candidates = np.array([[4., 4., .9], [4.2, 4., .8], [31., 0., .7]])
    tiles = [{"tile_id": 0, "image_xyxy": [0, 0, 32, 32], "ownership_xyxy": [0, 0, 32, 32]}]
    pred = TilePredictions(candidates, np.zeros(3, dtype=int), np.ones(3, dtype=bool), tiles,
                           np.array([[0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 7, 0]]))
    rec = count_identity_details(points, pred, .3)
    assert rec["unmatched_gt_ids"] == [1, 2]
    assert rec["false_positive_kind"] == ["duplicate_near_matched_gt", "background_outside_gt_radius"]
    final = TilePredictions(np.array([[20., 20., .91], [5., 4., .85]]), np.zeros(2, dtype=int),
                            np.ones(2, dtype=bool), tiles, np.array([[0, 5, 5, 0], [0, 1, 1, 1]]))
    changes = gt_transitions(rec, count_identity_details(points, final, .3))
    assert changes["recovered"] == [1] and changes["lost"] == []
    assert changes["unmatched_both"] == [2]
    kept, = changes["matched_both"]
    assert kept["gt_id"] == 0 and kept["displacement"] == 1. and kept["rank_change"] == 1
    assert kept["before_candidate_id"] == [0, 1, 1, 0] and kept["after_candidate_id"] == [0, 1, 1, 1]
