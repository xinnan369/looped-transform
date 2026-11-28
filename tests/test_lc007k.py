from copy import deepcopy
from pathlib import Path

import pytest

from scripts import archive_lc007i
from scripts.archive_lc007b import METRICS
from scripts.archive_lc007k import score
from scripts.prepare_lc007i import recipe, reference_tables
from scripts.prepare_lc007k import SCREEN, assert_paired_exposure, shared_control
from scripts.run_lc007d import read_json
from scripts.run_lc007i import EXPOSURE_FIELDS, expected_trajectory
from scripts.run_lc007k import authorize

PLAN = read_json(SCREEN)


def test_only_new_scope_and_independent_review_authorize_k():
    old = read_json(Path("reports/LC-007I/training_results/authorization.json"))
    with pytest.raises(ValueError):
        authorize(old, PLAN, "config", "head", "tree")
    record = {"issue": 36, "code_commit": "head", "source_tree": "tree", "config_sha256": "config",
              "limits": PLAN["limits_if_approved"], "owner_message_ref": "fixture", "approved_at": "fixture",
              "approval_scope": PLAN["scope"]}
    with pytest.raises(ValueError, match="independent review"):
        authorize(record | {"owner_review_waiver": "old"}, PLAN, "config", "head", "tree")
    authorize(record | {"independent_review": "fixture"}, PLAN, "config", "head", "tree")


def test_unshared_recipe_and_actual_exposure_are_bound():
    cfg = recipe(PLAN)
    assert cfg["variant"] == "shifted_unshared_4" and cfg["train_crop_size"] == 768
    assert cfg["tile_size"] == 512 and cfg["tile_overlap"] == 128 and cfg["seed"] == 42
    prior, evidence = shared_control(PLAN)
    assert evidence["case"]["successful_updates"] == 7200 and len(prior["training"]) == 40
    assert sum(int(row["valid_crop_pixels"]) for row in prior["training"]) == 8493465600
    copies = deepcopy(prior["training"])
    assert_paired_exposure(copies, prior["training"])
    copies[-1]["positive_slots"] = "123"
    with pytest.raises(ValueError, match="positive_slots"):
        assert_paired_exposure(copies, prior["training"])
    assert len(reference_tables(PLAN)["training"]) == 40


def test_expected_trajectory_uses_unshared_case_and_crop_only():
    row = dict.fromkeys(EXPOSURE_FIELDS, 1) | {"crop_size": 768, "crop_order_sha256": "crop",
           "learning_rates_sha256": "lr"}
    case = "42/U4-shift-768"
    frozen = {"planned_starts": {case: {"initial_sha256": "unshared"}},
              "exposure": [row | {"crop_size": 512}, row]}
    validate = expected_trajectory(frozen, 768, case)
    assert validate["initial_sha256"] == "unshared" and validate["trajectory"] == [("crop", "lr")]
    assert validate["crop_exposure"] == [dict.fromkeys(EXPOSURE_FIELDS, 1)]


def test_grid_preserves_loss_guard_even_when_raw_sharing_wins():
    def cell(mae, nap):
        common = dict.fromkeys(METRICS, 10.) | {"policy": "ema", "stage": "final", "density_group": "all"}
        return [common | {"epoch": e, "mae": mae, "local_nap": nap} for e in range(36, 41)]
    def training(late):
        return [{"epoch": e, "final_loss": late if e >= 36 else 1.} for e in range(31, 41)]
    s512 = {"metrics": cell(23.875, .30)}
    s768 = {"metrics": cell(19.7, .346), "training": training(.9587)}
    scored = score(training(1), cell(22, .325), cell(20.255, .3078), PLAN["decision"],
                   shared512=s512, shared768=s768)
    assert scored["decision"] == "fail"
    assert all(scored["sharing"]["raw"].values()) and all(scored["sharing"]["practical_margin"].values())
    assert not scored["sharing"]["guards"]["stability"] and not scored["sharing"]["joint_pass"]
    assert scored["difference_in_differences_shared_minus_unshared"]["mae"] == pytest.approx(-5.92)


def test_shared_archiver_keeps_partial_unshared_paths(tmp_path, monkeypatch):
    root = tmp_path / "run"
    case = root / "run/train/42/U4-shift-768"
    case.mkdir(parents=True)
    (case / "interrupted.pt").write_bytes(b"fixture")
    monkeypatch.setattr(archive_lc007i, "reference_tables", lambda plan: {})
    monkeypatch.setattr(archive_lc007i, "git", lambda *args: "tree")
    monkeypatch.setattr(archive_lc007i, "close_out", lambda *a, **kw: (
        {"code_commit": "head"}, {"execution": "stopped", "completed": []}, [], {"exit_status": 124}))
    out = tmp_path / "output"
    archive_lc007i.archive(root, out, PLAN, screen=SCREEN, authorize=lambda *args: None)
    assert read_json(out / "research_decision.json")["decision"] == "unevaluable"
    assert read_json(out / "run_evidence.json")["case"]["raw_files"][0]["bytes"] == len(b"fixture")
