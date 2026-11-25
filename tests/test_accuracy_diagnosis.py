import numpy as np
import pytest

from loopcrowd.accuracy_diagnosis import (
    aggregate,
    analyze_image,
    bootstrap_indices,
    density_bin,
    paired_rows,
    sampling_sensitivity,
)
from loopcrowd.calibration import candidate_digest
from loopcrowd.inference import TilePredictions
from loopcrowd.manifest import Record


def fixture(points):
    record = Record("test-fixture", "synthetic", "val", "unused.png", 64, 32, points)
    tiles = [{"tile_id": 0, "image_xyxy": [0, 0, 40, 32], "ownership_xyxy": [0, 0, 32, 32]},
             {"tile_id": 1, "image_xyxy": [24, 0, 64, 32], "ownership_xyxy": [32, 0, 64, 32]}]
    predictions = TilePredictions(np.array([[8.25, 8, .6], [32, 6, .3], [32, 6, .7], [55, 10, .8]]),
                                  np.array([0, 1, 0, 1]), np.array([True, True, False, True]), tiles)
    return record, predictions


def test_density_boundaries():
    assert [density_bin(x, (10, 20)) for x in (0, 10, 19, 20)] == ["low", "mid", "mid", "high"]


@pytest.mark.parametrize("points", [[], [[8, 8], [32, 6]]])
def test_image_metric_candidate_invariance_and_empty_gt(points):
    record, predictions = fixture(points)
    before = candidate_digest(predictions)
    row, ranked = analyze_image(record, predictions, (10, 20))
    assert row["predicted"] == 2
    assert (row["ap_candidates"], sum(row["score_hist_raw"])) == (4, 4)
    assert row["count_excluded_by_ownership"] == 1
    assert row["candidate_sha256"] == before == candidate_digest(predictions)
    assert row["tp"] == row["fn"] == bool(points)
    assert row["matched_px_mean"] == (.25 if points else None)
    assert row["covered_raw_floor"] == len(points)
    assert (aggregate([row], [ranked])["local_nap"] is None) == (not points)


def test_pairing_validates_full_image_identity():
    row, _ = analyze_image(*fixture([]), boundaries=(10, 20))
    assert paired_rows([row], [row])[0]["delta_absolute_error"] == 0
    for discarded in ([], [row | {"image_id": "different"}], [row | {"gt": 1}], [row, row]):
        with pytest.raises(ValueError, match="same ordered"):
            paired_rows([row], discarded)
    with pytest.raises(ValueError, match="unique"):
        paired_rows([row, row], [row, row])


def test_resampling_reuses_aligned_image_indices_and_separates_seed_dispersion():
    first = bootstrap_indices(3)
    second = bootstrap_indices(3)
    assert first.shape == (2000, 3)
    np.testing.assert_array_equal(first, second)
    a = sampling_sensitivity([[1, 2, 3], [3, 4, 5]], first)
    b = sampling_sensitivity([[3, 4, 5], [1, 2, 3]], second)
    assert a["per_seed_full_validation"] == [2, 4]
    assert a["leave_one_image_out"] == [3.5, 3, 2.5]
    assert a["bootstrap_p025_median_p975"] == b["bootstrap_p025_median_p975"]
    assert a["training_seed_sample_sd"] == pytest.approx(np.sqrt(2))
