import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loopcrowd.experiment import load_checkpoint, read_config, state_sha
from loopcrowd.inference import predict_image, predict_image_depth
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.model import LoopCrowd
from loopcrowd.training import fit
from scripts.prepare_lc007f import SCREEN
from scripts.run_lc007d import require_authorization
from scripts.run_lc007f import DepthEightBudget


PLAN = json.loads(SCREEN.read_text(encoding="utf-8"))
CONFIG = Path(PLAN["arm_config"])
STAGES = ("stage1", "stage2", "stage3", "early", "stage5", "stage6", "stage7", "final")


@pytest.mark.parametrize("arm", PLAN["arms"])
def test_eight_and_two_reads_agree_in_one_tiled_sweep_per_mode(arm):
    variant = PLAN["arms"][arm]["variant"]
    model = LoopCrowd(variant).eval()
    config = read_config(CONFIG, variant) | {"tile_size": 32, "tile_overlap": 8, "precision": "fp32"}
    blocks, decoded = [], []
    handles = [block.register_forward_hook(lambda module, args, result: blocks.append(id(module)))
               for block in model.blocks]
    handles.append(model.point_logits.register_forward_hook(lambda *args: decoded.append(1)))
    device = torch.device("cpu")
    before, rng = state_sha(model), torch.get_rng_state()
    with Image.new("RGB", (31, 29), (40, 110, 75)) as image:
        two, norms2 = predict_image_depth(model, image, config, device, diagnostic=False)
        assert len(blocks) == 8 and len(decoded) == 2 and len(set(blocks)) == len(model.blocks)
        eight, norms8 = predict_image_depth(model, image, config, device)
        assert len(blocks) == 16 and len(decoded) == 10
        deployment = predict_image(model, image, config, device)
        assert len(blocks) == 24 and len(decoded) == 11
    assert tuple(two) == ("early", "final") and tuple(eight) == STAGES
    assert norms8.shape == (1, 8, 2) and np.isfinite(norms8).all()
    np.testing.assert_array_equal(norms2, norms8[:, [3, 7], :])
    for stage in two:
        for field in ("candidates", "tile_ids", "owned"):
            np.testing.assert_array_equal(getattr(two[stage], field), getattr(eight[stage], field))
    np.testing.assert_array_equal(deployment.candidates, eight["final"].candidates)
    assert state_sha(model) == before and torch.equal(torch.get_rng_state(), rng) and not model.training
    for handle in handles:
        handle.remove()
    with pytest.raises(ValueError, match="all eight"):
        model(torch.zeros(1, 3, 32, 32), loop_count=4)


def test_schedule_changes_only_diagnostics_and_preserves_training_and_ema(tmp_path):
    manifest = tmp_path / "fixture.jsonl"
    Image.new("RGB", (32, 32), (100, 80, 25)).save(tmp_path / "tile.png")
    write_manifest(manifest, [Record(str(i), "synthetic", split, "tile.png", 32, 32, [[8.25, 13.75]])
                              for i, split in enumerate(("train", "train", "val"))])
    config = read_config(CONFIG, "shared_8_channel_residual_state") | {
        "pretrained": False, "tile_size": 32, "tile_overlap": 8, "precision": "fp32",
        "epochs": 2, "workers": 0, "lr_schedule": "constant",
    }
    device = torch.device("cpu")
    saved, rng = [], []
    for start in (1, 2):
        output = tmp_path / f"start-{start}"
        fit(config | {"depth_readout_start_epoch": start}, manifest, output, device)
        saved.append(load_checkpoint(output / "last.pt", device))
        rng.append(torch.get_rng_state())
    for field in ("model", "optimizer", "ema"):
        torch.testing.assert_close(saved[0][field], saved[1][field], atol=0, rtol=0)
    assert saved[0]["scheduler"] == saved[1]["scheduler"]
    assert torch.equal(*rng)
    for epoch, (all_stages, scheduled) in enumerate(zip(*(state["history"] for state in saved), strict=True), 1):
        for field in ("crop_order_sha256", "learning_rates", "gradient_l2", "early_loss", "final_loss",
                      "channel_gate_statistics"):
            assert all_stages[field] == scheduled[field]
        for policy in ("depth_live_validation", "ema_validation"):
            assert tuple(scheduled[policy]) == (("early", "final") if epoch == 1 else STAGES)
            assert all(value == all_stages[policy][stage] for stage, value in scheduled[policy].items())


def test_depth8_refuses_missing_authorization_and_budget(tmp_path):
    with pytest.raises(ValueError, match="owner authorization"):
        require_authorization({}, PLAN, "digest", "head")
    tiny = DepthEightBudget(PLAN["limits_if_approved"] | {"gpu_seconds": 20}, tmp_path / "ledger.jsonl",
                            torch.device("cpu"))
    with pytest.raises(RuntimeError, match="LC-007F time budget"):
        tiny.check("validation_image")
