from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import torch

from loopcrowd.calibration import json_digest
from loopcrowd.experiment import file_sha, make_model, read_config
from loopcrowd.grouped_validation import summarize
from loopcrowd.inference import TilePredictions, predict_image_depth
from scripts.run_lc007h_e40 import (
    METRIC_KEYS, e40_rows, gain_alignment, group_parity, image_parity, reconstruct, rng_digest,
    sample, summarize_bins,
)
from scripts.run_lc007d import read_json, require_authorization

SCREEN = read_json(Path("configs/lc007h_e40_diagnostic.json"))


def tiny_config():
    return read_config(Path("configs/lc007e_depth4.json"), "shifted_shared_4") | {
        "precision": "fp32", "tile_size": 32, "tile_overlap": 8,
    }


def test_model_reconstructs_only_ema_parameters_and_live_buffers(tmp_path):
    config = tiny_config()
    source = make_model(config, initialize=False)
    shadow = {name: torch.full_like(tensor, .125) for name, tensor in source.named_parameters()}
    live = source.state_dict()
    for name in live:
        if name.endswith("running_mean"):
            live[name].fill_(.5)
    row = {"epoch": 40, "ema_step": 7200, "ema_validation": {}}
    path = tmp_path / "last.pt"
    payload = {"model": live, "ema": {"step": 7200, "decay": config["ema_decay"], "shadow": shadow},
               "config": config, "data_sha256": "manifest", "git_commit": "source", "global_step": 7200,
               "history": [row] * 40, "optimizer": {"unused": [1]}}
    torch.save(payload, path)
    meta = {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size,
            "epoch": 40, "ema_step": 7200, "config_sha256": json_digest(config)}
    rng = rng_digest(torch.device("cpu"))
    model, metrics = reconstruct(meta, config, "manifest", "source", SCREEN)
    assert not model.training and rng_digest(torch.device("cpu")) == rng and metrics == {}
    assert all(torch.equal(tensor, shadow[name]) for name, tensor in model.named_parameters())
    assert all(torch.equal(tensor, live[name]) for name, tensor in model.named_buffers())
    assert meta["sha256"] == file_sha(path)
    with pytest.raises(ValueError, match="provenance"):
        reconstruct(meta, config, "manifest", "wrong_source", SCREEN)
    with pytest.raises(ValueError, match="frozen file"):
        reconstruct(meta | {"sha256": "bad"}, config, "manifest", "source", SCREEN)


def test_image_and_group_parity_fail_closed_preserving_rank_and_count():
    points = [[2., 3.]]
    values = np.array([[2., 3., .8], [11., 4., .8], [9., 8., .3]])
    prediction = TilePredictions(values, np.zeros(3, dtype=int), np.ones(3, dtype=bool), [],
                                np.array([[0, 0, 0, 1], [0, 1, 2, 0], [0, 2, 2, 0]]))
    metric, ranks, details = image_parity(points, prediction, .30)
    assert metric["predicted"] == 2 and metric["local_tp"] == metric["local_fp"] == 1
    assert details["candidate_ids"][:, -1].tolist() == [1, 0]
    rows = [{"gt": 1, "matches": metric}]
    saved = summarize(rows, [ranks])
    archive = dict(zip(METRIC_KEYS, (1, 1, 2, 1, 1., 1., 1., 1, 1, 0), strict=True))
    assert group_parity(rows, [ranks], archive, saved, 1e-12)["local_nap"] == 1.
    with pytest.raises(ValueError, match="grouped fp differs"):
        group_parity(rows, [ranks], archive | {"fp": 2}, saved, 1e-12)
    tampered = deepcopy(saved)
    tampered["nap_local_v1"]["ap_by_delta"]["0.05"] = .9
    with pytest.raises(ValueError, match="threshold AP differs"):
        group_parity(rows, [ranks], archive, tampered, 1e-12)


def test_synthetic_single_sweep_includes_every_gt_and_stops_on_archive_drift():
    config = tiny_config()
    model = make_model(config, initialize=False).eval()
    device = torch.device("cpu")
    screen = SCREEN | {"gt_rows_per_case": 2}
    record = SimpleNamespace(points=[[0., 0.], [38., 28.]], image_id="image")
    with Image.new("RGB", (39, 29)) as image:
        outputs, _ = predict_image_depth(model, image, config, device, export_identity=True)
        archived = {}
        for stage, predictions in outputs.items():
            counts, _, _ = image_parity(record.points, predictions, .30)
            archived[stage, "image"] = counts | {"gt": 2, "density_group": "low"}
        original = deepcopy(archived)
        decodes = []
        with model.point_logits.register_forward_hook(lambda *args: decodes.append(1)):
            _, _, info = sample(model, image, record, config, device, "low", screen, archived)
        assert not archived and len(decodes) == 4 * len(outputs["final"].tiles)
        assert info["unscorable_gt"] == 0 and sum(b["gt"] for b in info["bins"]) == 2
        assert all(row["fn_rate"] is None for row in summarize_bins([info], "high", 3))
        assert "stage3_to_final" in info and "gt_transitions" in info["midpoint"]
        original["final", "image"]["predicted"] += 1
        with pytest.raises(ValueError, match="image parity"):
            sample(model, image, record, config, device, "low", screen, original)


def test_empty_placebo_variation_duplicate_archive_and_scope_mismatch():
    maps = [(np.ones((1, 2), np.float32), np.ones((1, 2), bool))]
    assert gain_alignment(maps) == {"cells": 2, "changed": 0, "min": 1., "max": 1., "mean": 1., "sd": 0., "pearson": None}
    row = {"epoch": "40", "policy": "ema", "stage": "final"}
    with pytest.raises(ValueError, match="duplicate"):
        e40_rows([row, row], 40, ("stage",))
    authorization = {"owner_message_ref": "user", "approved_at": "time", "approval_scope": SCREEN["scope"],
                     "issue": 31, "code_commit": "sha", "config_sha256": "config", "limits": SCREEN["limits_if_approved"]}
    require_authorization(authorization, SCREEN, "config", "sha")
    with pytest.raises(ValueError, match="does not match"):
        require_authorization(authorization, SCREEN, "changed_config", "sha")
