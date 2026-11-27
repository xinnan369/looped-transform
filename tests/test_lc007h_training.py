import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.experiment import file_sha, load_checkpoint, make_model
from loopcrowd.grouped_validation import GainStatistics
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.training import fit
from scripts import archive_lc007h_training as archival
from scripts import run_lc007h_training as runner
from scripts.archive_lc007b import GROUPS, METRICS
from scripts.archive_lc007c import SHARED, UNSHARED
from scripts.run_lc007d import read_json

SCREEN = read_json(runner.SCREEN)
DEVICE = torch.device("cpu")


def test_training_binds_separate_scope_config_code_and_source_tree():
    owner = {"owner_message_ref": "test", "approved_at": "test", "approval_scope": SCREEN["scope"],
             "issue": 31, "config_sha256": "digest", "code_commit": "head", "source_tree": "tree",
             "limits": SCREEN["limits_if_approved"]}
    runner.require_training_authorization(owner, SCREEN, "digest", "head", "tree")
    for change in ({"config_sha256": "mismatch"}, {"code_commit": "elsewhere"}, {"source_tree": "mismatch"},
                   {"limits": SCREEN["limits_if_approved"] | {"total_updates": 43199}}):
        with pytest.raises(ValueError, match="does not match|source tree"):
            runner.require_training_authorization(owner | change, SCREEN, "digest", "head", "tree")
    old = read_json(Path("reports/LC-007H/e40_authorization.json"))
    with pytest.raises(ValueError):
        runner.require_training_authorization(old, SCREEN, "digest", "head", "tree")


def test_training_budget_stops_on_time_updates_and_phase_failure(tmp_path):
    ledger = tmp_path / "phases.jsonl"
    limits = SCREEN["limits_if_approved"]
    with pytest.raises(ValueError, match="without a quota"):
        runner.TrainingBudget(limits | {"peak_reserved_bytes": 1}, ledger, DEVICE)
    budget = runner.TrainingBudget(limits | {"total_updates": 1}, ledger, DEVICE)
    budget.check("train_batch")
    budget.record_update()
    with pytest.raises(RuntimeError, match="update cap"):
        budget.check("train_batch")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with budget.phase("validation_ema"):
            raise torch.cuda.OutOfMemoryError("synthetic")
    assert json.loads(ledger.read_text().splitlines()[0])["failure"] == "OutOfMemoryError: synthetic"
    budget.start -= limits["gpu_seconds"]
    with pytest.raises(RuntimeError, match="time budget"):
        budget.check("validation_image")


def test_runner_stops_after_first_oom_and_preserves_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda *args: ({
        "resource_projection": {"plan_seconds": 1, "required_device_headroom_bytes_estimate": 1},
        "planned_starts": {"42/S-guide": {"initial_sha256": "synthetic", "reference": "S4-shift"}},
    }, {(42, "S4-shift", "training"): [{
        "crop_order_sha256": "crop", "learning_rates_sha256": "lr",
    }]}))
    monkeypatch.setattr(runner, "background_snapshot", lambda: "synthetic")
    monkeypatch.setattr(runner, "git", lambda *args: "head" if args[0] == "rev-parse" else "")
    for operation, response in (("device_count", 1), ("get_device_name", "RTX 3090"), ("is_bf16_supported", True),
                                ("mem_get_info", (64, 128)), ("max_memory_allocated", 2),
                                ("max_memory_reserved", 3), ("synchronize", None)):
        monkeypatch.setattr(torch.cuda, operation, lambda *args, response=response: response)
    def fail_on_quota(*args):
        pytest.fail("quota must never be set")
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fail_on_quota)
    calls = []
    def fail(config, manifest, output, device, **kwargs):
        calls.append(output)
        raise torch.cuda.OutOfMemoryError("synthetic")
    monkeypatch.setattr(runner, "fit", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        runner.execute(tmp_path / "manifest", tmp_path / "run", SCREEN | {"disk_free_bytes": 1}, "head", "tree")
    progress = read_json(tmp_path / "run" / "queue_progress.json")
    assert len(calls) == 1 and progress["completed"] == []
    assert progress["execution"] == "stopped" and progress["current"] == "42/S-guide"
    assert progress["peak_reserved_bytes"] == 3 and "OutOfMemoryError" in progress["reason"]


@pytest.mark.parametrize("arm", ["S-guide", "S-placebo"])
def test_synthetic_fit_uses_two_training_and_four_validation_decodes_and_archives(tmp_path, monkeypatch, arm):
    manifest = tmp_path / "fixture.jsonl"
    Image.new("RGB", (32, 32), (100, 80, 25)).save(tmp_path / "tile.png")
    write_manifest(manifest, [Record(str(i), "synthetic", split, "tile.png", 32, 32, [[8.25, 13.75]])
                              for i, split in enumerate(("train", "train", "val"))])
    config = runner.training_config(SCREEN, 42, arm) | {
        "pretrained": False, "tile_size": 32, "tile_overlap": 8, "precision": "fp32",
        "epochs": 1, "workers": 0, "lr_schedule": "constant",
    }
    calls = []
    def traced(cfg, initialize):
        model = make_model(cfg, initialize)
        model.point_logits.register_forward_hook(lambda module, args, out: calls.append(model.training))
        return model
    monkeypatch.setattr("loopcrowd.training.make_model", traced)
    output = tmp_path / "case"
    run = fit(config, manifest, output, DEVICE)
    assert calls == [True] * 2 + [False] * 8
    assert run["global_step"] == 1 and run["completed_epochs"] == 1
    checkpoint = load_checkpoint(output / "last.pt", DEVICE)
    index, = read_json(output / "checkpoint_index.json")
    history = read_json(output / "history.json")
    assert index["sha256"] == file_sha(output / "last.pt") and index["step"] == checkpoint["ema"]["step"] == 1
    assert history[0]["image_metrics_sha256"] == file_sha(output / "images" / "01.json")
    assert history[0]["gradient_l2"]["nonfinite"] == 0 and history[0]["final_loss"] > 0
    assert len({row["stage"] for row in read_json(output / "images" / "01.json")}) == 4
    rows = list(archival.gain_rows(history, 42, arm))
    assert len(rows) == 8 and [row["valid_tokens"] for row in rows if row["density_group"] == "all"] == [16, 16]


def test_gain_statistics_define_empty_constant_and_phase_inverse():
    empty, constant, direct, inverse = (GainStatistics() for _ in range(4))
    assert empty.result() is None
    constant.add(np.ones((2, 4)), 17, 9)
    assert constant.result()["valid_tokens"] == 6
    assert constant.result()["aligned_permuted_pearson"] is None
    original = np.array([[.95, 1., 1.02, 1.04, .99]])
    direct.add(original, 40, 8)
    inverse.add(np.roll(original, 2), 40, 8)
    assert direct.result()["changed_fraction"] == inverse.result()["changed_fraction"] == 1
    assert direct.result()["aligned_permuted_pearson"] == pytest.approx(inverse.result()["aligned_permuted_pearson"])


def test_archive_partial_retains_hashes_and_marks_decision_unevaluable(tmp_path, monkeypatch):
    root = tmp_path / "launch"
    case = root / "run" / "train" / "42" / "S-guide"
    case.mkdir(parents=True)
    raw = case / "interrupted.pt"
    raw.write_bytes(b"partial evidence")
    monkeypatch.setattr(archival, "close_out", lambda *a, **k: (
        {"code_commit": "head"}, {"execution": "stopped", "completed": []}, [], {"exit_status": 124}))
    monkeypatch.setattr(archival, "git", lambda *a: "tree")
    monkeypatch.setattr(archival, "require_training_authorization", lambda *a: None)
    monkeypatch.setattr(archival, "verify_inputs", lambda *a: None)
    monkeypatch.setattr(archival, "archives", lambda: {})
    output = tmp_path / "archive"
    archival.archive(root, output, SCREEN)
    decision = read_json(output / "research_decision.json")
    evidence = read_json(output / "run_evidence.json")
    assert decision["status"] == "BLOCKED / PARTIAL_EVIDENCE_ARCHIVED" and decision["verified_runs"] == 0
    assert evidence["cases"][0]["raw_files"][0]["sha256"] == file_sha(raw)
    assert all(gate["decision"] == "unevaluable" for gate in decision["gates"].values())


def test_joint_decisions_separate_specificity_sharing_and_strict_paired_direction():
    arms = ("S-guide", "U-guide", "S-placebo", "S4-shift", "U4-shift", SHARED, UNSHARED)
    readouts, stable = {}, {}
    for seed in (42, 123):
        for arm in arms:
            mae = {"S-guide": 19, "U-guide": 20, "S-placebo": 21}.get(arm, 23)
            nap = {"S-guide": .33, "U-guide": .32, "S-placebo": .33 if seed == 42 else .31}.get(arm, .30)
            values = dict.fromkeys(METRICS, 10.) | {"mae": mae, "local_nap": nap}
            readouts[seed, arm] = [{"epoch": epoch, "policy": policy, "stage": "final", "density_group": group, **values}
                                   for epoch in SCREEN["decision"]["epochs"] for policy in ("live", "ema") for group in GROUPS]
            stable[seed, arm] = {"passed": True}
    _, checks, targets = archival.decisions(readouts, stable, SCREEN["decision"], True)
    assert checks["S-guide_vs_S4-shift"]["decision"] == "pass"
    assert checks["U-guide_vs_U4-shift"]["decision"] == "pass"
    assert checks["S-guide_vs_U-guide"]["decision"] == "pass"
    assert checks["S-guide_vs_S-placebo"]["decision"] == "fail"
    assert not checks["S-guide_vs_S-placebo"]["checks"]["paired_directions"]
    assert targets["S2"]["parity_mae_pass"]
