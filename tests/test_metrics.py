import math

import pytest
import torch
from PIL import Image

from loopcrowd.inference import decode, deduplicate, predict_image, tile_axis
from loopcrowd.metrics import count_metrics, point_nap


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


def test_nap_strict_threshold_and_no_cross_image_matches():
    gt = [[0, 0], [10, 0]]
    result = point_nap([(gt, [[5, 0, 1]])], thresholds=(0.5, 0.51))
    assert result["ap_by_delta"] == {"0.5": 0, "0.51": 0.5}
    assert point_nap([([[0, 0]], []), ([], [[0, 0, .9]])])["nap"] == 0
    assert point_nap([([], [])])["nap"] is None
    assert point_nap([([[1, 1], [1, 1]], [[1, 1, .8]])])["nap"] == 0.5


def test_tile_ownership_is_complete_and_dedup_uses_best_score():
    axes = tile_axis(897, 512, 128)
    assert axes[0][1] == 0 and axes[-1][2] == 897
    for (origin, start, end), (_, next_start, _) in zip(axes, axes[1:]):
        assert origin <= start < end <= origin + 512
        assert end == next_start
    assert deduplicate([[3, 4, .2], [4, 4, .8], [12, 20, .6]], 2) == [
        [4, 4, .8], [12, 20, .6],
    ]


def test_decode_restores_native_crop_coordinates():
    logits = torch.full((1, 2, 3, 4), -10.0)
    logits[0, 1, 1, 2] = 10
    offsets = torch.full((1, 2, 2, 3, 4), 0.25)
    assert decode({"logits": logits, "offsets": offsets}, 0.5)[0][:2] == [9.0, 5.0]


def test_full_frame_stitching_suppresses_duplicate_across_seam():
    class FixedTiles(torch.nn.Module):
        stride = 4

        def __init__(self):
            super().__init__()
            self.tiles = iter(((4, 0.875, 5), (3, 0.125, 6)))

        def forward(self, image):
            column, x_offset, score = next(self.tiles)
            logits = torch.full((1, 1, 8, 8), -10.0)
            logits[0, 0, 2, column] = score
            offsets = torch.full((1, 1, 2, 8, 8), 0.5)
            offsets[0, 0, 0, 2, column] = x_offset
            return {"logits": logits, "offsets": offsets}

    config = {"tile_size": 32, "tile_overlap": 8, "precision": "fp32",
              "localization_min_score": 0.05, "dedup_radius": 2}
    image = Image.new("RGB", (40, 32))
    predictions = predict_image(FixedTiles(), image, config, torch.device("cpu"))
    assert len(predictions) == 1 and predictions[0][:2] == [20.5, 10.0]
