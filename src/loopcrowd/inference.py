"""Full-frame tiled point inference with ownership and seam deduplication."""

from collections import defaultdict
from contextlib import nullcontext
from math import floor

import torch

from .data import image_tile


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def tile_axis(length, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("tile size must be positive and overlap must be in [0, size)")
    last = max(0, length - size)
    origins = list(range(0, last + 1, size - overlap))
    if origins[-1] != last:
        origins.append(last)
    boundaries = [0] + [(left + size + right) / 2 for left, right in zip(origins, origins[1:])] + [length]
    return list(zip(origins, boundaries, boundaries[1:]))


def decode(prediction, cutoff, stride=4):
    scores = prediction["logits"][0].float().sigmoid()
    slots, rows, cols = torch.where(scores > cutoff)
    offsets = prediction["offsets"][0, slots, :, rows, cols].float()
    x = (cols + offsets[:, 0]) * stride
    y = (rows + offsets[:, 1]) * stride
    return torch.stack((x, y, scores[slots, rows, cols]), dim=1).cpu().tolist()


def deduplicate(predictions, radius):
    """Score-ordered suppression in native coordinates with stable tie order."""
    if radius < 0:
        raise ValueError("deduplication radius cannot be negative")
    ranked = sorted(predictions, key=lambda row: (-row[2], row[0], row[1]))
    if radius == 0:
        return ranked
    cells = defaultdict(list)
    retained = []
    for x, y, score in ranked:
        column, row = floor(x / radius), floor(y / radius)
        if any(
            (x - px) ** 2 + (y - py) ** 2 <= radius**2
            for i in range(column - 1, column + 2)
            for j in range(row - 1, row + 2)
            for px, py in cells.get((i, j), ())
        ):
            continue
        cells[column, row].append((x, y))
        retained.append([x, y, score])
    return retained


@torch.inference_mode()
def predict_image(model, image, config, device):
    size, overlap = config["tile_size"], config["tile_overlap"]
    predictions = []
    for top, y_min, y_max in tile_axis(image.height, size, overlap):
        for left, x_min, x_max in tile_axis(image.width, size, overlap):
            tile, _, _ = image_tile(image, left, top, size)
            with autocast(device, config["precision"]):
                output = model(tile[None].to(device))
            for x, y, score in decode(output, config["localization_min_score"], model.stride):
                x, y = x + left, y + top
                if x_min <= x < x_max and y_min <= y < y_max:
                    predictions.append([x, y, score])
    return deduplicate(predictions, config["dedup_radius"])
