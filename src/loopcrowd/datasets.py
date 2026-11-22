"""Adapters for released annotation layouts. Coordinates are never clipped."""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .manifest import Record, ordered_records


def _points_array(value):
    array = np.asarray(value)
    if array.size == 0 and array.shape in ((0,), (0, 0), (0, 2)):
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind not in "iuf":
        raise ValueError("expected a numeric N×2 point matrix")
    return array.astype(float)


def _shanghai_points(path):
    try:
        # Keep the original matrix shape when extracting location from the
        # scalar MATLAB cell/struct. The metadata also contains the head count.
        info = loadmat(path, struct_as_record=False)["image_info"]
        while isinstance(info, np.ndarray) and info.size == 1:
            info = info.item()
        points = _points_array(info.location)
        if np.asarray(info.number).size != 1 or np.asarray(info.number).item() != len(points):
            raise ValueError("image_info.number differs from the number of locations")
        return points
    except (AttributeError, KeyError, TypeError, NotImplementedError, OSError, ValueError) as error:
        raise ValueError(f"{path}: invalid ShanghaiTech image_info: {error}") from error


def _nwpu_points(path, image_id):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not {"img_id", "human_num", "points"} <= data.keys():
            raise ValueError("expected official NWPU JSON with img_id, human_num and points")
        if data["img_id"] not in (f"{image_id}.jpg", f"{image_id}.jpeg"):
            raise ValueError(f"img_id {data['img_id']!r} does not match split ID {image_id}")
        raw_points = data["points"]
        if not isinstance(raw_points, list) or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(value) not in (int, float) for value in point)
            for point in raw_points
        ):
            raise ValueError("points must contain numeric [x, y] pairs")
        points = _points_array(raw_points)
        if type(data["human_num"]) is not int or data["human_num"] != len(points):
            raise ValueError("human_num differs from the number of points")
        return points
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid or unsupported NWPU JSON: {error}") from error


def _record(config, image, image_id, split, points, rejected_points):
    with Image.open(image) as source:
        width, height = source.size
    if points is not None:
        points = points - config["coordinate_base"]
        if rejected_points is not None and np.isfinite(points).all():
            valid = ((points >= 0) & (points < [width, height])).all(axis=1)
            rejected_points.extend(
                {
                    "dataset": config["dataset"],
                    "official_split": split,
                    "image_id": image_id,
                    "point_index": int(index),
                    "point_xy": points[index].tolist(),
                }
                for index in np.flatnonzero(~valid)
            )
            points = points[valid]
    try:
        return Record(
            image_id, config["dataset"], split, str(image.resolve()), width, height,
            None if points is None else points.tolist(),
        )
    except ValueError as error:
        raise ValueError(f"{image}: {error}") from error


def _by_stem(directory, extension):
    if not directory.is_dir():
        raise ValueError(f"missing directory {directory}")
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() == extension:
            if path.stem in files:
                raise ValueError(f"{directory}: duplicate file ID {path.stem}")
            files[path.stem] = path
    return files


def _shanghai(config, root, rejected_points):
    records = []
    for split in ("train", "test"):
        folder = root / config["split_dirs"][split]
        images = _by_stem(folder / "images", config["image_suffix"])
        raw_annotations = _by_stem(folder / "ground_truth", ".mat")
        if not images:
            raise ValueError(f"{folder}: no {config['image_suffix']} images")
        for stem, image in images.items():
            annotation = raw_annotations.get(f"GT_{stem}")
            if annotation is None:
                raise ValueError(f"{image}: missing ground_truth/GT_{stem}.mat")
            # Official split folders reuse IMG numbers.
            records.append(
                _record(config, image, f"{split}/{stem}", split, _shanghai_points(annotation),
                        rejected_points)
            )
        for stem, path in raw_annotations.items():
            if not stem.startswith("GT_") or stem[3:] not in images:
                raise ValueError(f"{path}: missing corresponding image")

    validation = config["validation"]
    fraction = validation["fraction"]
    seed = validation["seed"]
    if type(fraction) not in (int, float) or not 0 <= fraction < 1 or type(seed) is not int:
        raise ValueError("validation requires a fraction in [0, 1) and an integer seed")
    train_ids = [record.image_id for record in records if record.split == "train"]
    ranked = sorted(
        train_ids,
        key=lambda image_id: (hashlib.sha256(f"{seed}:{image_id}".encode()).digest(), image_id),
    )
    val_ids = set(ranked[:int(len(ranked) * fraction)])
    return [replace(record, split="val") if record.image_id in val_ids else record for record in records]


def _nwpu_split(path, split):
    ids = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            valid_lengths = (1, 3) if split == "test" else (3,)
            if (
                len(fields) not in valid_lengths
                or not re.fullmatch(r"[0-9]{4}", fields[0])
                or any(not re.fullmatch(r"[0-9]+", value) for value in fields[1:])
            ):
                raise ValueError(f"{path}:{number}: unsupported official split row")
            ids.append(fields[0])
    if not ids:
        raise ValueError(f"{path}: empty split")
    return ids


def _nwpu(config, root, rejected_points):
    entries = {}
    for split in ("train", "val", "test"):
        for image_id in _nwpu_split(root / config["split_files"][split], split):
            if image_id in entries:
                raise ValueError(f"duplicate image ID {image_id} in NWPU official split files")
            entries[image_id] = split

    records = []
    for image_id, split in entries.items():
        image = root / config["images"] / f"{image_id}{config['image_suffix']}"
        points = None
        if split != "test":
            annotation = root / config["annotations"] / f"{image_id}.json"
            points = _nwpu_points(annotation, image_id)
        records.append(_record(config, image, image_id, split, points, rejected_points))
    return records


def build_manifest(config_path: Path, root: Path, rejected_points=None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        if type(config["coordinate_base"]) is not int or config["coordinate_base"] not in (0, 1):
            raise ValueError("coordinate_base must be explicitly set to 0 or 1")
        if config["image_suffix"] not in (".jpg", ".jpeg"):
            raise ValueError("image_suffix must be .jpg or .jpeg")
        if config["format"] == "shanghaitech_mat_v5":
            records = _shanghai(config, root, rejected_points)
        elif config["format"] == "nwpu_official_points_json":
            records = _nwpu(config, root, rejected_points)
        else:
            raise ValueError(f"unsupported annotation format {config['format']!r}")
    except (KeyError, TypeError) as error:
        raise ValueError(f"{config_path}: invalid dataset configuration: {error}") from error
    return ordered_records(records)
