#!/usr/bin/env python3
"""Measure a complete batch step and batch-one inference on the selected hardware."""

import argparse
import sys
import statistics
import time
from pathlib import Path

# Keep scripts/profile.py from shadowing Python's profile module via cProfile.
if __name__ == "__main__" and sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import numpy as np
import torch
from torch import nn

from loopcrowd.data import image_tile, load_rgb
from loopcrowd.experiment import make_model, provenance, read_config, seed_all, write_json
from loopcrowd.inference import autocast, predict_image
from loopcrowd.manifest import read_manifest
from loopcrowd.model import VARIANTS
from loopcrowd.targets import point_targets
from loopcrowd.training import gpu_scaler, optimizer_for, train_step


def forward_flops(model, image):
    counts = []

    def count(module, arguments, result):
        if isinstance(module, nn.Conv2d):
            counts.append(2 * result.numel() * module.in_channels // module.groups
                          * module.kernel_size[0] * module.kernel_size[1])
        elif isinstance(module, nn.Linear):
            counts.append(2 * result.numel() * module.in_features)
        else:
            windows, tokens, width = arguments[0].shape
            counts.append(windows * (8 * tokens * width**2 + 4 * tokens**2 * width))

    handles = [module.register_forward_hook(count) for module in model.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention))]
    try:
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    return sum(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Use one real validation image for batch-one latency")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    config = read_config(args.config, args.variant)
    device, size = torch.device(args.device), config["tile_size"]
    use_cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if use_cuda else lambda: None
    seed_all(config["seed"])
    model = make_model(config, initialize=True).to(device)
    optimizer = optimizer_for(model, config)
    scaler = gpu_scaler(device, config["precision"])
    images = torch.randn(config["batch_size"], 3, size, size, device=device)
    points = np.array([[1.5, 2.5], [8.1, 9.5], [size - 0.1, size - 0.1]])
    target = {k: torch.stack([v] * config["batch_size"]).to(device)
              for k, v in point_targets(points, size, size, config["slots"]).items()}

    model.eval()
    flops = forward_flops(model, images[:1])
    model.train()
    train_step(model, optimizer, scaler, images, target, config, device)
    sync()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    def measure(call):
        timings = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            result = call()
            sync()
            timings.append(1000 * (time.perf_counter() - start))
        return {"samples_ms": timings, "median_ms": statistics.median(timings)}, result

    steps, loss = measure(lambda: train_step(model, optimizer, scaler, images, target, config, device))
    if not torch.isfinite(loss):
        raise RuntimeError("profiled step loss was non-finite")
    peak_allocated = torch.cuda.max_memory_allocated(device) if use_cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if use_cuda else None
    model.eval()
    full_image = None
    sample_size = None
    sample = images[:1]
    if args.manifest:
        record = next(record for record in read_manifest(args.manifest) if record.split == "val")
        with load_rgb(record, args.manifest.parent) as image:
            sample_size = list(image.size)
            sample = image_tile(image, 0, 0, size)[0][None].to(device)
            predict_image(model, image, config, device)
            full_image, _ = measure(lambda: predict_image(model, image, config, device))

    @torch.inference_mode()
    def inference():
        with autocast(device, config["precision"]):
            return model(sample)

    inference()
    latency, logits = measure(inference)
    args.output.mkdir(parents=True)
    run = provenance(config, device, args.manifest)
    run["execution"] = "completed"
    run["weight_initialization"]["loaded"] = config["pretrained"]
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "forward_flops_per_image": flops,
        "flops_method": "2 per MAC, conv+linear+MHA projections and attention matmuls",
        "train_batch": config["batch_size"],
        "train_step": steps, "batch1_forward": latency, "batch1_full_image": full_image,
        "batch1_full_image_size": sample_size,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "bf16_supported": torch.cuda.is_bf16_supported() if use_cuda else None,
        "fp16_supported": torch.cuda.get_device_capability(device) >= (5, 3) if use_cuda else None,
        "last_train_loss": float(loss),
        "output_shape": list(logits["logits"].shape),
    }
    write_json(args.output / "run_manifest.json", run)
    write_json(args.output / "profile.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
