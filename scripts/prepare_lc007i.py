"""Verify frozen native B data, crop trajectories, and single-run resource estimates."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from loopcrowd.calibration import json_digest
from loopcrowd.crops import crop_points
from loopcrowd.data import TrainingCrops, load_rgb
from loopcrowd.experiment import file_sha, make_model, seed_all
from loopcrowd.manifest import read_manifest
from loopcrowd.training import UpdateSchedule
from scripts.archive_lc007c import check_training
from scripts.diagnose_lc007h import G
from scripts.prepare_lc007d import initialization
from scripts.prepare_lc007e import config_for
from scripts.profile import forward_costs
from scripts.run_lc007a import read_csv
from scripts.run_lc007b import frozen_inputs
from scripts.run_lc007d import read_json

SCREEN = Path("configs/lc007i_native_crop.json")
SEED, ARM, REF = 42, "S4-shift-768", "S4-shift"


def recipe(plan):
    arm, = plan["order"][str(SEED)]
    return config_for(plan, SEED, arm) | {"train_crop_size": plan["train_crop_size"]}


def reference_tables(plan):
    for name, path in (("base_config", Path(plan["arm_config"])), ("archive_index", G / "epoch_index.csv"),
                       ("reference_audit", G / "initialization_audit.json"),
                       ("reference_evidence", G / "run_evidence.json"), ("phase", G / "gpu_phases.jsonl")):
        if file_sha(path) != plan[f"{name}_sha256"]:
            raise ValueError(f"frozen source differs: {path}")
    tables = {}
    for row in read_csv(G / "epoch_index.csv"):
        if (row["seed"], row["arm"]) == (str(SEED), plan.get("historical_reference", REF)):
            path = G / row["path"]
            if path.stat().st_size != int(row["bytes"]) or file_sha(path) != row["sha256"]:
                raise ValueError(f"invalid historical table: {path}")
            tables[row["kind"]] = read_csv(path)
            if len(tables[row["kind"]]) != int(row["rows"]):
                raise ValueError(f"historical row count differs: {path}")
    check_training(tables["training"])
    return tables


def exposure(records, directory, config):
    steps = (len(records) + config["batch_size"] - 1) // config["batch_size"]
    schedule = UpdateSchedule(SimpleNamespace(param_groups=[{}]), config, steps * config["epochs"])
    if steps != 180 or schedule.warmup_updates != 360:
        raise ValueError("historical update schedule differs")
    points = [np.asarray(r.points, dtype=float).reshape(-1, 2) for r in records]
    size = config.get("train_crop_size", config["tile_size"])
    stride = 4
    for epoch in range(1, config["epochs"] + 1):
        crops = TrainingCrops(records, directory, config, epoch)
        sampler = DataLoader(range(len(crops)), batch_size=config["batch_size"], shuffle=True,
                             generator=torch.Generator().manual_seed(config["seed"] + epoch))
        order, images = hashlib.sha256(), hashlib.sha256()
        rates = []
        pixels = heads = overflow = valid_slots = 0
        for batch in sampler:
            identities = []
            for index in batch.tolist():
                r = records[index]
                left, top = crops.origin(r, points[index])
                width, height = min(size, r.image_width - left), min(size, r.image_height - top)
                h = crop_points(points[index], left, top, size)
                if min(width, height) <= 0 or (len(h) and not (h < [width, height]).all()):
                    raise ValueError("unsupported native crop geometry")
                cols, rows = (width + stride - 1) // stride, (height + stride - 1) // stride
                cells = (h // stride).astype(int)
                counts = np.bincount(cells[:, 1] * cols + cells[:, 0])
                overflow += int(np.maximum(counts - config["slots"], 0).sum())
                heads += len(h)
                pixels += width * height
                valid_slots += config["slots"] * cols * rows
                identities.append((index, left, top, size) if "train_crop_size" in config else (index, left, top))
            order.update(np.asarray(identities, dtype="<i8").tobytes())
            images.update(batch.numpy().astype("<i8", copy=False).tobytes())
            rates.append(schedule.apply())
            schedule.global_step += 1
        yield {"epoch": epoch, "crop_size": size, "crop_order_sha256": order.hexdigest(),
               "image_order_sha256": images.hexdigest(), "learning_rates_sha256": json_digest(rates),
               "valid_crop_pixels": pixels, "padding_pixels": len(crops) * size**2 - pixels,
               "sampled_gt": heads, "overflow_gt": overflow, "positive_slots": heads - overflow,
               "negative_slots": valid_slots - (heads - overflow)}


def audit(manifest, plan, *, screen=SCREEN):
    arm, = plan["order"][str(SEED)]
    reference = plan.get("historical_reference", REF)
    hist = reference_tables(plan)
    _, data = frozen_inputs(manifest)
    records = read_manifest(manifest, splits=("train", "val"))
    geometry = {}
    for split in ("train", "val"):
        selected = [r for r in records if r.split == split]
        for r in selected:
            with load_rgb(r, manifest.parent):
                pass
        geometry[split] = {"images": len(selected), "min_width": min(r.image_width for r in selected),
                           "min_height": min(r.image_height for r in selected),
                           "max_width": max(r.image_width for r in selected),
                           "max_height": max(r.image_height for r in selected)}
    train = [r for r in records if r.split == "train"]
    config = recipe(plan)
    baseline = {key: value for key, value in config.items() if key != "train_crop_size"}
    historical = read_json(G / "initialization_audit.json")
    prior = historical["planned_starts"][f"{SEED}/{reference}"]
    if json_digest(baseline) != prior["config_sha256"] or any(
        data[k] != historical["data"][k] for k in ("manifest_sha256", "pretrained_weight_sha256",
                                                   "ordered_val_ids", "train_only_density_tertiles")
    ):
        raise ValueError("frozen historical recipe or split differs")
    small, large = (list(exposure(train, manifest.parent, cfg)) for cfg in (baseline, config))
    for a, b, c in zip(small, large, hist["training"], strict=True):
        if (a["crop_order_sha256"] != c["crop_order_sha256"]
                or a["learning_rates_sha256"] != b["learning_rates_sha256"]
                or a["learning_rates_sha256"] != c["learning_rates_sha256"]
                or a["image_order_sha256"] != b["image_order_sha256"]
                or any(a[k] != int(c[k]) for k in ("sampled_gt", "overflow_gt", "positive_slots", "negative_slots"))):
            raise ValueError(f"historical or paired data trajectory differs at epoch {a['epoch']}")
    seed_all(SEED)
    model = make_model(config, initialize=True).eval()
    initial = initialization(model)
    if any(value != prior[key] for key, value in initial.items()):
        raise ValueError("model initialization, parameters or RNG differs")
    costs = {str(size): forward_costs(model, torch.zeros(1, 3, size, size), depth_midpoint=True)
             for size in (512, 768)}
    evaluation = forward_costs(model, torch.zeros(1, 3, 512, 512), depth_staged=True)
    with (G / "gpu_phases.jsonl").open(encoding="utf-8") as stream:
        phases = [json.loads(line) for line in stream]
    case = [p for p in phases if p["case"] == f"{SEED}/{reference}"]
    phase_times = {phase: sum(p["seconds"] for p in case if p["phase"] == phase)
                   for phase in ("training", "validation_live", "validation_ema", "initialization", "checkpoint")}
    area = (768 / 512)**2
    dense_ratio = costs["768"]["dense_flops"] / costs["512"]["dense_flops"]
    projected = sum(phase_times.values()) + phase_times["training"] * (max(area, dense_ratio) - 1)
    return {"plan_sha256": file_sha(screen), "data": data, "decoded_geometry": geometry,
            "planned_starts": {f"{SEED}/{arm}": initial | {"config_sha256": json_digest(config)}},
            "historical_config_sha256": prior["config_sha256"], "exposure": small + large,
            "dense_two_read_flops": costs, "dense_four_read_512_flops": evaluation,
            "projection": {"historical_phase_seconds": phase_times, "area_ratio": area, "dense_ratio": dense_ratio,
                           "base_seconds": projected, "allowance_seconds": projected * .25 + 120,
                           "budget_estimate_seconds": projected * 1.25 + 120,
                           "headroom_bytes_estimate": int(2 * max(p["peak_reserved_bytes"] for p in case) * max(area, dense_ratio)),
                           "actual_gpu": "not_run"}}
