"""Deterministic slot assignment and losses shared by all model variants."""

import numpy as np
import torch
from torch.nn import functional as F


def point_targets(points, width, height, slots=2, stride=4, valid_width=None, valid_height=None):
    columns, rows = (width + stride - 1) // stride, (height + stride - 1) // stride
    labels = np.zeros((slots, rows, columns), dtype=np.float32)
    offsets = np.zeros((slots, 2, rows, columns), dtype=np.float32)
    counts = np.zeros((rows, columns), dtype=np.int64)
    valid_width = width if valid_width is None else valid_width
    valid_height = height if valid_height is None else valid_height
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not (((points >= 0) & (points < [valid_width, valid_height])).all()):
        raise ValueError("target points must be finite and within the valid image region")
    for x, y in sorted(map(tuple, points)):
        column, row = int(x // stride), int(y // stride)
        slot = counts[row, column]
        counts[row, column] += 1
        if slot < slots:
            labels[slot, row, column] = 1
            offsets[slot, :, row, column] = x / stride - column, y / stride - row

    valid = ((np.arange(rows)[:, None] * stride < valid_height)
             & (np.arange(columns)[None, :] * stride < valid_width))
    return {
        "labels": torch.from_numpy(labels),
        "offsets": torch.from_numpy(offsets),
        "valid": torch.from_numpy(valid[None]),
        "density": torch.from_numpy(counts[None].astype(np.float32)),
        "heads": torch.tensor(len(points)),
        "overflow": torch.tensor(np.maximum(counts - slots, 0).sum()),
    }


def point_loss(prediction, target, density_weight=0, components=None):
    logits, labels = prediction["logits"].float(), target["labels"]
    valid, positive = target["valid"], labels.bool()
    scores = logits.sigmoid()
    focal = torch.where(positive, 0.25 * (1 - scores).square(), 0.75 * scores.square())
    normalizer = labels.sum().clamp_min(1)
    classification = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                      * focal * valid).sum() / normalizer
    offset_error = F.smooth_l1_loss(prediction["offsets"].float(), target["offsets"], reduction="none")
    regression = (offset_error * positive.unsqueeze(2)).sum() / normalizer
    if components is not None:
        components.update(classification_loss=classification.detach(), offset_loss=regression.detach())
    loss = classification + regression
    if density_weight:
        density_error = F.smooth_l1_loss(prediction["density"].float(), target["density"], reduction="none")
        loss = loss + density_weight * (density_error * valid).sum() / valid.sum().clamp_min(1)
    return loss
