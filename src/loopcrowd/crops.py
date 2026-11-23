"""Reproducible native-scale random and density-biased crops."""

import hashlib

import numpy as np


def crop_points(points, left, top, size):
    shifted = points - [left, top]
    return shifted[((shifted >= 0) & (shifted < size)).all(axis=1)]


def crop_origin(points, width, height, size, mode, seed, identity, epoch=0):
    token = hashlib.sha256(f"{seed}:{identity}:{epoch}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:16], "big"))
    limits = max(0, width - size), max(0, height - size)

    def random_origin():
        return tuple(int(rng.integers(limit + 1)) for limit in limits)

    origin = random_origin()
    if mode == "random" or not len(points):
        return origin
    if mode != "density-biased":
        raise ValueError(f"unsupported crop policy {mode!r}")
    # Compare the uniform draw and three point-centered, randomly jittered proposals.
    candidates = [origin]
    for index in rng.integers(len(points), size=3):
        center = points[index] - rng.uniform(0.25, 0.75, size=2) * size
        candidates.append(tuple(int(value) for value in np.clip(np.floor(center), 0, limits)))
    return max(candidates, key=lambda candidate: len(crop_points(points, *candidate, size)))
