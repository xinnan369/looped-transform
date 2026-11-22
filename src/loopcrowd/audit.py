"""Full-image and 512-pixel crop capacity statistics."""

import hashlib
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree


STRIDES = (4, 8)
CAPACITIES = (1, 2, 4, 8)
QUANTILES = (0, 25, 50, 75, 90, 95, 99, 100)
CROP_SIZE = 512


def _quantiles(values):
    if len(values) == 0:
        return None
    return {
        f"p{q}": float(value)
        for q, value in zip(QUANTILES, np.percentile(values, QUANTILES))
    }


def _neighbor_distances(points):
    if len(points) < 2:
        return np.empty(0)
    locations, counts = np.unique(points, axis=0, return_counts=True)
    distances = cKDTree(locations).query(locations, k=2)[0][:, 1]
    distances[counts > 1] = 0
    return np.repeat(distances, counts)


def _grid_histogram(points, width, height, stride):
    columns = (width + stride - 1) // stride
    rows = (height + stride - 1) // stride
    cells = np.floor(points / stride).astype(np.int64)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    histogram = Counter(counts.tolist())
    histogram[0] = columns * rows - len(counts)
    return histogram


def _grid_summary(histogram, head_count):
    total = sum(histogram.values())
    occupied = total - histogram[0]
    overflow = {}
    for capacity in CAPACITIES:
        lost = sum(max(0, points - capacity) * count for points, count in histogram.items())
        overflow[str(capacity)] = {
            "lost_points": lost,
            "point_loss_rate": lost / head_count if head_count else None,
        }
    return {
        "total_cells": total,
        "occupied_cells": occupied,
        "occupied_fraction": occupied / total,
        "max_points_per_cell": max(histogram),
        "cell_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "overflow": overflow,
    }


def _sample(record, mode, seed):
    points = np.asarray(record.points, dtype=float).reshape(-1, 2)
    width, height = record.image_width, record.image_height
    if mode == "full-image":
        return points, width, height, None

    # Fixed per-image offsets are independent of manifest order and filtering.
    token = hashlib.sha256(f"{seed}:{record.dataset}:{record.image_id}".encode()).digest()
    left = int.from_bytes(token[:16], "big") % (max(0, width - CROP_SIZE) + 1)
    top = int.from_bytes(token[16:], "big") % (max(0, height - CROP_SIZE) + 1)
    shifted = points - [left, top]
    keep = ((shifted >= 0) & (shifted < CROP_SIZE)).all(axis=1)
    return shifted[keep], CROP_SIZE, CROP_SIZE, [left, top]


def _audit_group(records, mode, seed):
    images = []
    neighbors = []
    histograms = {stride: Counter() for stride in STRIDES}
    for record in records:
        points, width, height, origin = _sample(record, mode, seed)
        distances = _neighbor_distances(points)
        if len(distances):
            neighbors.append(distances)
        grids = {}
        for stride in STRIDES:
            histogram = _grid_histogram(points, width, height, stride)
            histograms[stride].update(histogram)
            grids[str(stride)] = _grid_summary(histogram, len(points))
        images.append({
            "image_id": record.image_id,
            "width": width,
            "height": height,
            "crop_origin_xy": origin,
            "source_head_count": len(record.points),
            "head_count": len(points),
            "nearest_neighbor_px": _quantiles(distances),
            "grids": grids,
        })

    counts = [image["head_count"] for image in images]
    return {
        "image_count": len(images),
        "source_head_count": sum(image["source_head_count"] for image in images),
        "head_count": sum(counts),
        "width_px": _quantiles([image["width"] for image in images]),
        "height_px": _quantiles([image["height"] for image in images]),
        "heads_per_image": _quantiles(counts),
        "nearest_neighbor_px": _quantiles(np.concatenate(neighbors) if neighbors else []),
        "grids": {
            str(stride): _grid_summary(histogram, sum(counts))
            for stride, histogram in histograms.items()
        },
        "images": images,
    }


def make_audit(records, source, modes, seed, split=None, include_images=True):
    groups = defaultdict(list)
    skipped = Counter()
    for record in records:
        if split is not None and record.split != split:
            continue
        key = record.dataset, record.split
        if record.points is None:
            skipped[key] += 1
        else:
            groups[key].append(record)
    if not groups:
        raise ValueError("no labeled images for the requested split")

    summaries = []
    for dataset in sorted({dataset for dataset, _ in groups}):
        dataset_groups = {
            split_name: group for (name, split_name), group in groups.items() if name == dataset
        }
        if len(dataset_groups) > 1:
            dataset_groups["all"] = [
                record for group in dataset_groups.values() for record in group
            ]
        for split_name, group in sorted(dataset_groups.items()):
            ordered = sorted(group, key=lambda record: record.key)
            for mode in modes:
                summary = _audit_group(ordered, mode, seed)
                if not include_images or split_name == "all":
                    del summary["images"]
                summaries.append({"dataset": dataset, "split": split_name, "mode": mode, **summary})
    return {
        "schema_version": 1,
        "source": source,
        "crop_size_px": CROP_SIZE,
        "crop_seed": seed,
        "unlabeled": [
            {"dataset": dataset, "split": split_name, "image_count": count}
            for (dataset, split_name), count in sorted(skipped.items())
        ],
        "groups": summaries,
    }


def _percent(rate):
    return "—" if rate is None else f"{rate:.4%}"


def _median(summary):
    return "—" if summary is None else f"{summary['p50']:.2f}"


def to_markdown(report):
    rows = [
        f"# Point-candidate capacity audit ({report['source']})",
        "",
        f"Crop configuration uses one {CROP_SIZE}×{CROP_SIZE} crop per image "
        f"with seed {report['crop_seed']}.",
        "Counts and rates in the table are pooled within each dataset, split and mode.",
        "The nearest-neighbor median pools within-image point distances.",
        "Point loss rates divide lost points by the number of points in that row.",
        "",
        "| Dataset | Split | Mode | Images | Source heads | Audited heads | Median W × H | "
        "Median heads/image | Median NN (px) | Stride | Occupied cells | K=1 | K=2 | K=4 | K=8 |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        for stride, grid in group["grids"].items():
            losses = " | ".join(
                _percent(grid["overflow"][str(k)]["point_loss_rate"]) for k in CAPACITIES
            )
            size = f"{_median(group['width_px'])} × {_median(group['height_px'])}"
            rows.append(
                f"| {group['dataset']} | {group['split']} | {group['mode']} | "
                f"{group['image_count']} | {group['source_head_count']} | {group['head_count']} | "
                f"{size} | {_median(group['heads_per_image'])} | "
                f"{_median(group['nearest_neighbor_px'])} | {stride} | "
                f"{_percent(grid['occupied_fraction'])} | {losses} |"
            )
    if report["unlabeled"]:
        rows += ["", "Excluded unlabeled images", ""]
        rows.extend(
            f"- {item['dataset']} / {item['split']}: {item['image_count']} images"
            for item in report["unlabeled"]
        )
    return "\n".join(rows) + "\n"
