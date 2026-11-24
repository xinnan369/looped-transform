import json

import pytest
import torch
from PIL import Image

from loopcrowd.experiment import DEFAULT_CONFIG, load_checkpoint
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.scheduling import UpdateSchedule
from loopcrowd.training import fit


def test_update_schedule_hits_fixed_horizon_endpoints():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0002)
    schedule = UpdateSchedule(optimizer, DEFAULT_CONFIG | {"lr_schedule": "warmup_cosine"}, 5400)
    rates = []
    for _ in range(5400):
        rates.append(schedule.apply())
        assert optimizer.param_groups[0]["lr"] == rates[-1]
        assert schedule.apply() == rates[-1]
        schedule.global_step += 1
    assert schedule.warmup_updates == 270
    assert rates[0] == rates[-1] == 0.00002
    assert rates[269] == rates[270] == 0.0002
    assert rates[:270] == sorted(rates[:270])
    assert rates[270:] == sorted(rates[270:], reverse=True)
    with pytest.raises(ValueError, match="horizon"):
        schedule.apply()


def test_warmup_cosine_resume_keeps_update_trajectory(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    image = tmp_path / "fixture.png"
    Image.new("RGB", (35, 43), (100, 50, 20)).save(image)
    rows = [Record(str(index), "fixture", split, image.name, 35, 43, [[8, 12]])
            for index, split in enumerate(("train", "train", "train", "train", "val"))]
    write_manifest(manifest, rows)
    config = DEFAULT_CONFIG | {
        "epochs": 3, "tile_size": 32, "tile_overlap": 8,
        "lr_schedule": "warmup_cosine", "fixed_training_horizon": True,
    }
    device = torch.device("cpu")
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    fit(config, manifest, full, device)
    partial = fit(config, manifest, interrupted, device, stop_after=1)
    assert partial["execution"] == "paused" and partial["global_step"] == 2
    with pytest.raises(ValueError, match="fixed training horizon"):
        fit(config | {"epochs": 4}, manifest, interrupted, device, resume=True)
    fit(config, manifest, interrupted, device, resume=True)
    a, b = [load_checkpoint(path / "last.pt", device) for path in (full, interrupted)]
    assert a["global_step"] == b["global_step"] == 6
    for key in ("scheduler", "scaler", "best_epoch"):
        assert a[key] == b[key]
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    for key, state in a["optimizer"]["state"].items():
        assert all(torch.equal(value, b["optimizer"]["state"][key][field]) for field, value in state.items())
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    histories = [json.loads((path / "history.json").read_text()) for path in (full, interrupted)]
    assert [epoch["learning_rates"] for epoch in histories[0]] == [
        epoch["learning_rates"] for epoch in histories[1]
    ]
