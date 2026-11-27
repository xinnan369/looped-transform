"""Detached midpoint confidence heuristic for a complete residual update."""

import torch
from torch.nn import functional as F

REFINEMENT_MODES = ("guided", "placebo")


def midpoint_uncertainty(logits, valid=None):
    probability = logits.detach().float().sigmoid()
    score = (4 * probability * (1 - probability)).amax(dim=1, keepdim=True)
    if valid is None:
        valid = torch.ones_like(score, dtype=torch.bool)
    if valid.shape != score.shape:
        raise ValueError("point-grid validity must match the midpoint logits")
    score = F.max_pool2d(score.masked_fill(~valid, 0), 2, 2, ceil_mode=True)
    valid = F.max_pool2d(valid.float(), 2, 2, ceil_mode=True).bool()
    return score[:, 0], valid[:, 0]


def refinement_gain(logits, valid=None, *, mode="guided"):
    score, support = midpoint_uncertainty(logits, valid)
    return gain_from_uncertainty(score, support, mode=mode)


def gain_from_uncertainty(score, support, *, mode):
    """Construct or permute the gains from an aligned midpoint map."""
    # Subtracting an anchor first preserves exact centering for constant maps.
    relative = (score - score.amax(dim=(-2, -1), keepdim=True)).masked_fill(~support, 0)
    mean = relative.sum(dim=(-2, -1), keepdim=True) / support.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    gain = torch.where(support, 1 + .25 * (relative - mean), 1.)
    if mode == "placebo":
        values = []
        for sample, mask in zip(gain, support, strict=True):
            shifted = torch.ones_like(sample)
            active = sample[mask]
            shifted[mask] = active.roll(active.numel() // 2)
            values.append(shifted)
        return torch.stack(values)
    return gain


def refine_update(before, after, gain):
    if gain.shape != before.shape[:-1]:
        raise ValueError("midpoint confidence grid must match the P3 tokens")
    scale = gain.unsqueeze(-1)
    updated = before.float() + scale * (after.float() - before.float())
    return torch.where(scale == 1, after, updated.to(after.dtype))
