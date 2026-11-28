from copy import deepcopy

import pytest

from scripts.audit_lc007j import POLICIES, STAGES, average, pack, paired, record, summary


def make(image, *, gt, tp, fp):
    signed = fp - (gt - tp)
    return {"image_id": image, "density_group": "low", "gt": gt, "predicted": tp + fp,
            "local_tp": tp, "local_fp": fp, "local_fn": gt - tp, "count_error": signed,
            "abs_count_error": abs(signed), "canceled_unmatched": 2 * min(fp, gt - tp)}


def test_reject_bad_matching_and_cancellation():
    valid = make("A", gt=10, tp=6, fp=3)
    assert record(valid)["canceled_unmatched"] == 6
    with pytest.raises(ValueError, match="identity"):
        record(valid | {"canceled_unmatched": 9})
    with pytest.raises(ValueError, match="identity"):
        record(valid | {"local_tp": 7})


def test_precision_recall_denominators_and_undefined_values():
    stats = summary([record(make("A", gt=10, tp=5, fp=3)), record(make("B", gt=1, tp=1, fp=3))])
    assert (stats["precision"], stats["recall"]) == (6 / 12, 6 / 11)
    assert stats["over_magnitude"] == 3 and stats["under_magnitude"] == 2
    empty = summary([record(make("A", gt=0, tp=0, fp=0))])
    assert empty["precision"] is None and empty["recall"] is None
    assert average([stats, empty])["precision"] is None


def test_pair_delta_decomposes_signed_error():
    before = record(make("A", gt=7, tp=4, fp=2))
    after = record(make("A", gt=7, tp=6, fp=4))
    change = paired(before, after)
    assert (change["delta_count_error"], change["delta_local_fp"], change["delta_local_fn"]) == (4, 2, -2)
    assert change["delta_abs_count_error"] == 2


def test_pack_rejects_missing_duplicate_and_reordered_images():
    entries = [make(i, gt=7, tp=4, fp=2) | {"epoch": e, "policy": p, "stage": s}
               for e in range(1, 41) for p in POLICIES for s in STAGES for i in ("A", "B")]
    assert len(pack(entries, ["A", "B"], {})) == 320
    with pytest.raises(ValueError, match="ordered"):
        pack(entries[1:], ["A", "B"], {})
    with pytest.raises(ValueError, match="duplicate"):
        pack(entries + [deepcopy(entries[0])], ["A", "B"], {})
    with pytest.raises(ValueError, match="ordered"):
        pack([entries[1], entries[0], *entries[2:]], ["A", "B"], {})
