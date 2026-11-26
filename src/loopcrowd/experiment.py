"""Experiment configuration, provenance and checkpoint utilities."""

import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import ResNet18_Weights

from .ema import validate_decay
from .model import CHANNEL_GATES, DEPTH8_VARIANTS, DEPTH_VARIANTS, LoopCrowd, VARIANTS, WEIGHTS, pretrained_path

DEFAULT_CONFIG = {
    "variant": "cnn_only", "seed": 42, "slots": 2, "pretrained": False,
    "density_weight": 0, "loop_embedding": False, "gradient_checkpointing": False,
    "tile_size": 512, "tile_overlap": 128, "crop_mode": "random", "batch_size": 2,
    "workers": 0, "precision": "fp32", "epochs": 1, "patience": 0, "lr": 0.0002,
    "weight_decay": 0.01, "count_threshold": 0.30, "localization_min_score": 0.05,
    "stage": "smoke_only", "label_policy": "synthetic",
    "lr_schedule": "constant", "min_lr": 0.00002, "warmup_fraction": 0.05,
    "fixed_training_horizon": False,
    "backbone_bn": "train",
    "early_loss_weight": None,
    "audit_training": False,
}


def read_config(path, variant=None):
    overrides = json.loads(path.read_text(encoding="utf-8"))
    unknown = overrides.keys() - DEFAULT_CONFIG.keys() - {"ema_decay", "depth_screen", "depth_readout_start_epoch"}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    config = DEFAULT_CONFIG | overrides
    if variant is not None:
        config["variant"] = variant
    if config["variant"] not in VARIANTS:
        raise ValueError("unknown model variant")
    early = config["early_loss_weight"]
    depth = config.get("depth_screen", False)
    if type(depth) is not bool or (depth and (
        config["variant"] not in DEPTH_VARIANTS or early != .25 or config["loop_embedding"]
        or config["density_weight"] or config["gradient_checkpointing"]
    )):
        raise ValueError("depth_screen requires a supported fixed-window variant with frozen midpoint supervision")
    if config["variant"] in DEPTH8_VARIANTS and depth:
        start = config.get("depth_readout_start_epoch")
        if type(start) is not int or type(config["epochs"]) is not int or not 1 <= start <= config["epochs"]:
            raise ValueError("K=8 depth readout start must be within the training horizon")
    elif "depth_readout_start_epoch" in config:
        raise ValueError("depth_readout_start_epoch requires the K=8 screen")
    two_call = ("shared_2", "unshared_2", *(name for name in CHANNEL_GATES if name not in DEPTH_VARIANTS))
    if early is not None and (
        (not depth and config["variant"] not in two_call) or type(early) not in (int, float)
        or not math.isfinite(early) or early < 0
    ):
        raise ValueError("early_loss_weight requires a supported fixed-window model and a finite nonnegative value")
    if type(config["audit_training"]) is not bool:
        raise ValueError("audit_training must be a boolean")
    if config.get("ema_decay") is not None:
        validate_decay(config["ema_decay"])
    for name in ("slots", "tile_size", "batch_size", "epochs"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("workers", "patience", "tile_overlap"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if config["tile_size"] < 16 or config["tile_overlap"] >= config["tile_size"]:
        raise ValueError("tile size must be at least 16 and greater than overlap")
    if not 0 <= config["localization_min_score"] <= config["count_threshold"] <= 1:
        raise ValueError("localization score floor must be no greater than the count threshold")
    if config["crop_mode"] not in ("random", "density-biased"):
        raise ValueError("unsupported crop policy")
    if config["precision"] not in ("fp32", "bf16", "fp16"):
        raise ValueError("unsupported precision")
    if config["backbone_bn"] not in ("train", "frozen"):
        raise ValueError("unsupported backbone batch normalization policy")
    if config["stage"] not in ("smoke_only", "pilot", "full_train"):
        raise ValueError("unsupported execution stage")
    if not config["label_policy"] or config["density_weight"] < 0:
        raise ValueError("set a label policy and nonnegative density weight")
    if config["lr_schedule"] not in ("constant", "warmup_cosine"):
        raise ValueError("unsupported learning-rate schedule")
    if type(config["fixed_training_horizon"]) is not bool:
        raise ValueError("fixed_training_horizon must be a boolean")
    if not math.isfinite(config["lr"]) or config["lr"] < 0:
        raise ValueError("lr must be finite and nonnegative")
    if config["lr_schedule"] == "warmup_cosine":
        if not 0 < config["min_lr"] <= config["lr"] or not 0 < config["warmup_fraction"] < 1:
            raise ValueError("warmup cosine requires 0 < min_lr <= lr and 0 < warmup_fraction < 1")
    if config["patience"] and (config["fixed_training_horizon"] or config["lr_schedule"] != "constant"):
        raise ValueError("fixed training horizons cannot use early stopping")
    return config


def make_model(config, initialize):
    return LoopCrowd(
        config["variant"], slots=config["slots"], pretrained=config["pretrained"] and initialize,
        density=config["density_weight"] > 0, loop_embedding=config["loop_embedding"],
        gradient_checkpointing=config["gradient_checkpointing"],
        freeze_backbone_bn=config.get("backbone_bn", "train") == "frozen",
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(model, *, prefixes=None):
    """Hash the exact tensor contents, including buffers, independent of checkpoint packaging."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if prefixes is not None and not name.startswith(prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), tuple(tensor.shape)], separators=(",", ":"))
        digest.update(header.encode("utf-8") + b"\n")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*args):
    process = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None


def weight_provenance(config, loaded):
    result = {"requested": WEIGHTS if config["pretrained"] else "random", "loaded": loaded}
    if loaded:
        url = ResNet18_Weights[WEIGHTS].url
        path = pretrained_path()
        result.update(url=url, cache_path=str(path.resolve()), sha256=file_sha(path))
    return result


def provenance(config, device, manifest=None):
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "tracked_changes": _git("status", "--porcelain", "--untracked-files=no"),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "seed": config["seed"],
        "status": config["stage"],
        "environment": {
            "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "data": None if manifest is None else {"manifest": str(manifest.resolve()), "sha256": file_sha(manifest)},
        "label_policy": config["label_policy"],
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "lr_schedule": config.get("lr_schedule", "constant"),
        "weight_initialization": {"requested": WEIGHTS if config["pretrained"] else "random"},
        "eval_protocol": {
            "inference": "native-scale tiles with source tile and midpoint ownership",
            "count": "ownership-only candidates, strict score cutoff, no radius suppression",
            "localization": "nap_local_v1 on all in-frame tile candidates above the score floor",
            "p2pnet_nap_parity": "unavailable_upstream_evaluator",
            "nwpu_official_f1": "not_ready",
        },
    }


def checkpoint(path, payload):
    partial = path.with_suffix(".tmp")
    torch.save(payload, partial)
    partial.replace(path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)
