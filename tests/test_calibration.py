from dataclasses import replace

import numpy as np
import pytest

from loopcrowd.calibration import (
    GRID_POINTS, candidate_digest, choose_fit_records, count_match_metrics, fit_cutoff, fixed_pool_ap, gap_summary,
    rank_metrics, remap_scores, strict_counts,
)
from loopcrowd.inference import TilePredictions
from loopcrowd.manifest import Record


def test_fit_selection_is_stratified_repeatable_and_train_only():
    def record(split, number):
        return Record(f"{split}-{number}", "fixture", split, "fixture.png", 100, 100,
                      [[float(i), 0.] for i in range(number)])

    records = [record(split, n) for split in ("train", "val", "test")
               for n in (3, 4, 5, 11, 12, 13, 21, 22, 23)]
    manifest, fit = choose_fit_records(records, (900, 1900), per_group=2)
    repeated, repeated_fit = choose_fit_records(list(reversed(records)), (900, 1900), per_group=2)
    assert manifest == repeated and fit == repeated_fit
    assert len(fit) == 6 and all(row.split == "train" for row in fit)
    assert all(len(rows) == 2 for rows in manifest.values())
    assert all(set(row) == {"image_id", "selection_sha256"} for rows in manifest.values() for row in rows)
    assert not {r.key for r in fit} & {r.key for r in records if r.split != "train"}


def test_strict_cutoffs_and_both_tie_breaks():
    scores = np.array([0.05, 0.30, np.float64(np.float32(0.30)), 0.95, 1.0])
    assert strict_counts(scores, [0.05, 0.30, 0.95]).tolist() == [4, 3, 1]
    errors = np.full(len(GRID_POINTS), 100)
    errors[GRID_POINTS.index(29)] = errors[GRID_POINTS.index(31)] = 99
    assert fit_cutoff(errors)[0] == 0.31
    errors[GRID_POINTS.index(30)] = 99
    assert fit_cutoff(errors)[0] == 0.30


def test_one_to_one_matches_use_the_count_prefix_and_keep_candidates():
    points = [[5., 5.], [40., 40.]]
    predictions = TilePredictions(
        candidates=np.array([[5., 5., .85], [5.2, 5., .85], [40., 40., .3], [40., 40., .95], [80., 80., .15]]),
        tile_ids=np.array([0, 0, 0, 1, 0]), owned=np.array([True, True, True, False, True]),
        tiles=[{"tile_id": 0}, {"tile_id": 1}],
    )
    before = candidate_digest(predictions)
    raw, lower = count_match_metrics(points, predictions.count_candidates(), (.30, .20))
    assert (raw["predicted"], raw["local_tp"], raw["local_fp"], raw["local_fn"]) == (2, 1, 1, 1)
    assert (lower["predicted"], lower["local_tp"], lower["local_fp"], lower["local_fn"]) == (3, 2, 1, 0)
    ranked = rank_metrics(points, predictions)
    first = fixed_pool_ap([ranked, ranked], 4, .20)
    assert first["original"] == first["count_calibrated_fixed_pool"] == first["remap_check"]
    assert first["remap_changed_rank_positions"] == first["remap_new_ties_between_distinct_scores"] == 0
    assert first["original_tie_positions"] > 0
    assert first == fixed_pool_ap([ranked, ranked], 4, .20)
    assert before == candidate_digest(predictions)
    assert before != candidate_digest(replace(predictions, tile_ids=np.zeros(5, dtype=int)))


@pytest.mark.parametrize("cutoff", (0.05, 0.30, 0.95))
def test_monotonic_odds_mapping_preserves_rank_and_boundaries(cutoff):
    scores = np.array([0., .05, .30, .30, .95, 0.99999994, 1.])
    mapped = remap_scores(scores, cutoff)
    np.testing.assert_array_equal(np.argsort(-scores, kind="stable"), np.argsort(-mapped, kind="stable"))
    assert mapped[0] == 0 and mapped[-1] == 1
    assert remap_scores([cutoff], cutoff)[0] == pytest.approx(.30)


def test_gap_gate_uses_three_seed_mean_and_two_positive_gaps():
    result = gap_summary([-5, 10, 15], [-8, 3, 4], True)
    assert result["per_seed_closure"] == [None, 0.7, 11 / 15]
    assert result["seeds_with_narrower_positive_raw_gap"] == 2 and result["pass"]
    assert not gap_summary([-5, 10, 15], [-8, 3, 4], False)["pass"]
