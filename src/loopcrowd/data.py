"""Native-scale image crops and deterministic training fixtures."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .crops import crop_origin, crop_points
from .targets import point_targets


def load_rgb(record, manifest_dir):
    path = manifest_dir / Path(record.image_path)
    with Image.open(path) as image:
        if image.size != (record.image_width, record.image_height):
            raise ValueError(f"image dimensions changed after manifest creation: {path}")
        return image.convert("RGB")


def image_tile(image, left, top, size):
    width = min(size, image.width - left)
    height = min(size, image.height - top)
    pixels = np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tile = np.zeros((3, size, size), dtype=np.float32)
    tile[:, :height, :width] = ((pixels - mean) / std).transpose(2, 0, 1)
    return torch.from_numpy(tile), width, height


class TrainingCrops(Dataset):
    def __init__(self, records, manifest_dir, config, epoch):
        self.records = records
        self.manifest_dir = manifest_dir
        self.config = config
        self.epoch = epoch
        self.size = config.get("train_crop_size", config["tile_size"])

    def __len__(self):
        return len(self.records)

    def origin(self, record, points):
        return crop_origin(
            points, record.image_width, record.image_height, self.size, self.config["crop_mode"],
            self.config["seed"], f"{record.dataset}:{record.image_id}", self.epoch,
        )

    def __getitem__(self, index):
        record = self.records[index]
        points = np.asarray(record.points, dtype=float).reshape(-1, 2)
        size = self.size
        left, top = self.origin(record, points)
        with load_rgb(record, self.manifest_dir) as image:
            tile, width, height = image_tile(image, left, top, size)
        target = point_targets(
            crop_points(points, left, top, size), size, size, self.config["slots"],
            valid_width=width, valid_height=height,
        )
        if self.config.get("early_loss_weight") is not None or self.config.get("audit_training"):
            identity = [index, left, top]
            if "train_crop_size" in self.config:
                identity.append(size)
                target["valid_pixels"] = torch.tensor(width * height)
            target["crop_identity"] = torch.tensor(identity)
        return tile, target
