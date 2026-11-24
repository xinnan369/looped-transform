import math

import pytest
import torch
from PIL import Image

from loopcrowd.inference import decode, predict_image, tile_axis
from loopcrowd.metrics import count_metrics, image_matches, point_nap, score_threshold_sweep


def test_count_metrics():
    result = count_metrics([0, 7], [1, 4])
    assert result == {"mae": 2.0, "rmse": math.sqrt(5)}


def test_nap_score_order_one_to_one_and_ap_envelope():
    gt = [[0, 0], [10, 0]]
    predictions = [[10, 0, .20], [0, 0, .50], [0, 0, .90], [99, 99, .99]]
    result = point_nap([(gt, predictions)], thresholds=(0.5,))
    # Descending rank gives FP, TP, duplicate FP, TP. Precision envelope is .5.
    assert result["nap"] == pytest.approx(0.5)
    assert result["ap_by_delta"] == {"0.5": pytest.approx(0.5)}


def test_low_score_input_row_cannot_claim_ground_truth_first():
    gt = [[0, 0]]
    predictions = [[0.1, 0, 0.1], [0, 0, 0.9]]
    ranked, matches = image_matches(gt, predictions, (0.5,))
    assert ranked.tolist() == [1, 0]
    assert matches.tolist() == [[0, -1]]
    assert point_nap([(gt, predictions)], (0.5,))["ap_by_delta"] == {"0.5": 1.0}


def test_nap_pools_predictions_in_global_score_order_without_cross_image_matches():
    samples = [
        ([[0, 0]], [[0, 0, 0.4]]),
        ([[4, 4]], [[4, 4, 0.8], [0, 0, 0.95]]),
    ]
    assert point_nap(samples, (0.5,))["nap"] == pytest.approx(2 / 3)


def test_equal_scores_preserve_input_and_image_order():
    gt = [[0, 0]]
    predictions = [[0.1, 0, 0.7], [0, 0, 0.7]]
    ranked, matches = image_matches(gt, predictions, (0.5,))
    assert ranked.tolist() == [0, 1]
    assert matches.tolist() == [[0, -1]]
    empty_image_fp = ([], [[0, 0, 0.7]])
    tp_image = (gt, [predictions[1]])
    assert point_nap([empty_image_fp, tp_image], (0.5,))["nap"] == 0.5
    assert point_nap([tp_image, empty_image_fp], (0.5,))["nap"] == 1


def test_sweep_true_positive_prefix_matches_the_retained_high_score_prediction():
    samples = [([[0, 0]], [[0.1, 0, 0.1], [0, 0, 0.9]])]
    rows = score_threshold_sweep(samples, (0.05, 0.5, 0.9))
    assert [row["predicted_heads"] for row in rows] == [2, 1, 0]
    assert [row["local_true_positives_delta_0_5"] for row in rows] == [1, 1, 0]
    assert [row["local_precision"] for row in rows] == [0.5, 1.0, None]
    assert [row["local_recall"] for row in rows] == [1.0, 1.0, 0]


def test_nap_strict_threshold_and_no_cross_image_matches():
    gt = [[0, 0], [10, 0]]
    result = point_nap([(gt, [[5, 0, 1]])], thresholds=(0.5, 0.51))
    assert result["ap_by_delta"] == {"0.5": 0, "0.51": 0.5}
    assert point_nap([([[0, 0]], []), ([], [[0, 0, .9]])])["nap"] == 0
    assert point_nap([([], [])])["nap"] is None
    assert point_nap([([[1, 1], [1, 1]], [[1, 1, .8]])])["nap"] == 0.5


def test_tile_ownership_is_complete():
    axes = tile_axis(897, 512, 128)
    assert axes[0][1] == 0 and axes[-1][2] == 897
    for (origin, start, end), (_, next_start, _) in zip(axes, axes[1:]):
        assert origin <= start < end <= origin + 512
        assert end == next_start


def test_decode_restores_native_crop_coordinates():
    logits = torch.full((1, 2, 3, 4), -10.0)
    logits[0, 1, 1, 2] = 10
    offsets = torch.full((1, 2, 2, 3, 4), 0.25)
    assert decode({"logits": logits, "offsets": offsets}, 0.5)[0, :2].tolist() == [9.0, 5.0]


class FixedTiles(torch.nn.Module):
    stride = 4

    def __init__(self, tiles):
        super().__init__()
        self.tiles = iter(tiles)

    def forward(self, image):
        logits = torch.full((1, 2, 8, 8), -10.0)
        offsets = torch.full((1, 2, 2, 8, 8), 0.5)
        for slot, (x, score) in enumerate(next(self.tiles)):
            column = int(x // 4)
            logits[0, slot, 2, column] = score
            offsets[0, slot, 0, 2, column] = x / 4 - column
        return {"logits": logits, "offsets": offsets}


def predict_fixture(width, tiles):
    config = {"tile_size": 32, "tile_overlap": 8, "precision": "fp32",
              "localization_min_score": 0.05}
    return predict_image(FixedTiles(tiles), Image.new("RGB", (width, 32)), config, torch.device("cpu"))


def test_two_real_close_heads_inside_a_tile_survive_both_candidate_paths():
    gt = [[8.5, 10], [9.5, 10]]
    result = predict_fixture(32, [[(8.5, 6), (9.5, 5)]])
    assert result.candidates[:, :2].tolist() == gt
    assert result.count_candidates()[:, :2].tolist() == gt
    assert point_nap([(gt, result.candidates)])["nap"] == 1


def test_repeated_overlap_point_is_owned_once_but_ap_sees_the_duplicate():
    result = predict_fixture(40, [[(22, 7)], [(14, 6)]])
    assert result.candidates[:, :2].tolist() == [[22, 10], [22, 10]]
    assert result.tile_ids.tolist() == [0, 1]
    assert result.owned.tolist() == [False, True]
    assert len(result.count_candidates()) == 1
    assert result.diagnostics(0.3)["count_excluded_by_ownership"] == 1
    assert image_matches([[22, 10]], result.candidates, (0.5,))[1].tolist() == [[0, -1]]


def test_two_real_close_heads_across_the_seam_both_survive():
    result = predict_fixture(40, [[(19.5, 6)], [(12.5, 7)]])
    gt = [[19.5, 10], [20.5, 10]]
    assert result.count_candidates()[:, :2].tolist() == gt
    assert result.diagnostics(0.3)["ambiguous_owned_pairs_within_2px"] == 1
    assert result.diagnostics(0.3)["reconciled_pairs"] == 0
    assert point_nap([(gt, result.candidates)])["nap"] == 1


def test_sweep_count_matching_and_strict_score_boundary():
    samples = [([[8, 8]], [[8, 8, .9], [8, 8, .5]]), ([], [[0, 0, .9]])]
    rows = score_threshold_sweep(samples, (0.49, 0.5, 0.9))
    assert [row["predicted_heads"] for row in rows] == [3, 2, 0]
    assert [row["local_precision"] for row in rows] == [1 / 3, 0.5, None]
    assert [row["local_recall"] for row in rows] == [1, 1, 0]
