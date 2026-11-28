import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.data import TrainingCrops
from loopcrowd.experiment import file_sha, make_model
from loopcrowd.inference import predict_image_depth
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.training import fit
from scripts import archive_lc007i as archival
from scripts.archive_lc007b import METRICS
from scripts.prepare_lc007i import SCREEN, recipe
from scripts.run_lc007d import read_json
from scripts.run_lc007h_training import require_training_authorization
from scripts.run_lc007i import NativeCropBudget, require_launch_record

PLAN = read_json(SCREEN)
DEVICE = torch.device("cpu")


def test_native_crop_preserves_pixels_padding_heads_and_legacy_identity(tmp_path):
    frame = np.full((29, 53, 3), [30, 80, 140], dtype=np.uint8)
    Image.fromarray(frame).save(tmp_path / "frame.png")
    row = Record("1", "fixture", "train", "frame.png", 53, 29, [[52., 28.], [52., 28.], [52., 28.]])
    config = recipe(PLAN) | {"tile_size": 32, "tile_overlap": 8, "train_crop_size": 64}
    images, target = TrainingCrops([row], tmp_path, config, 1)[0]
    assert images.shape == (3, 64, 64) and torch.count_nonzero(images[:, 29:]) == 0
    assert torch.count_nonzero(images[:, :, 53:]) == 0
    assert int(target["valid"].sum()) == 8 * 14
    assert (int(target["heads"]), int(target["overflow"])) == (3, 1)
    assert int(target["valid_pixels"]) == 29 * 53 and target["crop_identity"].tolist() == [0, 0, 0, 64]
    legacy = {k: v for k, v in config.items() if k != "train_crop_size"}
    old_tile, old_target = TrainingCrops([row], tmp_path, legacy, 1)[0]
    assert old_tile.shape == (3, 32, 32) and len(old_target["crop_identity"]) == 3
    assert "valid_pixels" not in old_target
    model = make_model(config, initialize=False).eval()
    with Image.open(tmp_path / "frame.png") as image:
        a, _ = predict_image_depth(model, image, legacy | {"precision": "fp32"}, DEVICE)
        b, _ = predict_image_depth(model, image, config | {"precision": "fp32"}, DEVICE)
    for stage in a:
        np.testing.assert_array_equal(a[stage].candidates, b[stage].candidates)
        assert a[stage].tiles == b[stage].tiles


def test_fit_logs_crop_size_and_uses_two_reads_with_independent_eval_geometry(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    write_manifest(manifest, [Record(str(i), "fixture", split, "image.png", 32, 32, [[8., 9.]])
                              for i, split in enumerate(("train", "train", "val"))])
    config = recipe(PLAN) | {"train_crop_size": 48, "tile_size": 32, "tile_overlap": 8, "pretrained": False,
                             "precision": "fp32", "epochs": 2, "workers": 0, "lr_schedule": "constant"}
    sizes, reads = [], []
    def traced(config, initialize):
        model = make_model(config, initialize)
        model.register_forward_pre_hook(lambda module, args: sizes.append(tuple(args[0].shape[-2:])))
        model.point_logits.register_forward_hook(lambda *args: reads.append(model.training))
        return model
    monkeypatch.setattr("loopcrowd.training.make_model", traced)
    output = tmp_path / "run"
    result = fit(config, manifest, output, DEVICE, stop_after=1)
    row, = read_json(output / "history.json")
    assert sizes == [(48, 48), (32, 32), (32, 32)]
    assert reads == [True] * 2 + [False] * 8
    assert result["global_step"] == row["ema_step"] == 1
    assert row["valid_crop_pixels"] == 2048 and row["sampled_gt"] == 2
    assert len(row["image_order_sha256"]) == len(row["crop_order_sha256"]) == 64
    legacy = {key: value for key, value in config.items() if key != "train_crop_size"}
    with pytest.raises(ValueError, match="resume requires matching"):
        fit(legacy, manifest, output, DEVICE, resume=True)


def test_distinct_scope_and_resource_stops(tmp_path):
    old = read_json(Path("reports/LC-007H/training_results/authorization.json"))
    with pytest.raises(ValueError):
        require_training_authorization(old, PLAN, file_sha(SCREEN), "head", "tree")
    limits = PLAN["limits_if_approved"]
    path = tmp_path / "ledger"
    with pytest.raises(ValueError, match="without a quota"):
        NativeCropBudget(limits | {"peak_reserved_bytes": 99}, path, DEVICE)
    budget = NativeCropBudget(limits, path, DEVICE)
    budget.updates = limits["total_updates"]
    with pytest.raises(RuntimeError, match="update cap"):
        budget.check("train_batch")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with budget.phase("validation_live"):
            raise torch.cuda.OutOfMemoryError("fixture")
    assert json.loads(path.read_text())["failure"] == "OutOfMemoryError: fixture"
    budget.start -= 1800
    with pytest.raises(RuntimeError, match="time budget"):
        budget.check("validation_live")


def test_launch_record_requires_review_or_explicit_owner_waiver():
    owner = {"issue": 33, "approval_scope": PLAN["scope"], "limits": PLAN["limits_if_approved"],
             "code_commit": "head", "config_sha256": "config", "source_tree": "tree",
             "owner_message_ref": "fixture", "approved_at": "fixture"}
    with pytest.raises(ValueError, match="explicit waiver"):
        require_launch_record(owner | {"owner_review_waiver": " "}, PLAN, "config", "head", "tree")
    for field in ("independent_review", "owner_review_waiver"):
        require_launch_record(owner | {field: "fixture evidence"}, PLAN, "config", "head", "tree")
    with pytest.raises(ValueError):
        require_launch_record(owner | {"owner_review_waiver": "fixture"}, PLAN, "other", "head", "tree")


def test_partial_archive_preserves_raw_and_disables_scoring(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    case = root / "run" / "train" / "42" / "S4-shift-768"
    case.mkdir(parents=True)
    path = case / "interrupted.pt"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(archival, "reference_tables", lambda plan: {})
    monkeypatch.setattr(archival, "require_launch_record", lambda *args: None)
    monkeypatch.setattr(archival, "git", lambda *args: "tree")
    monkeypatch.setattr(archival, "close_out", lambda *args, **kwargs: (
        {"code_commit": "head", "independent_review": "fixture"},
        {"execution": "stopped", "completed": []}, [], {"exit_status": 124}))
    dest = tmp_path / "archive"
    archival.archive(root, dest, PLAN)
    assert read_json(dest / "research_decision.json")["decision"] == "unevaluable"
    assert read_json(dest / "run_evidence.json")["case"]["raw_files"][0]["sha256"] == file_sha(path)


def test_joint_research_gate_keeps_strict_nap():
    rule = PLAN["decision"]
    training = [{"epoch": e, "final_loss": 1.} for e in range(31, 41)]
    common = dict.fromkeys(METRICS, 10.) | {"policy": "ema", "stage": "final", "density_group": "all"}
    a = [common | {"epoch": e, "mae": 23., "local_nap": .31} for e in rule["analysis_epochs"]]
    b = [row | {"mae": 23.875, "local_nap": rule["nap_reference"]} for row in a]
    assert archival.score(training, a, b, rule)["decision"] == "pass"
    tied = [row | {"local_nap": rule["nap_reference"]} for row in a]
    assert archival.score(training, tied, b, rule)["decision"] == "fail"
