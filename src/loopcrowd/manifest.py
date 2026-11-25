"""Shared manifest records and strict JSONL validation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True)
class Record:
    image_id: str
    dataset: str
    split: str
    image_path: str
    image_width: int
    image_height: int
    points: list[list[float]] | None

    def __post_init__(self):
        for value in (self.image_id, self.dataset, self.image_path):
            if not isinstance(value, str) or not value:
                raise ValueError("image_id, dataset and image_path must be nonempty strings")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"unsupported split {self.split!r}")
        if any(type(size) is not int or size < 1 for size in (self.image_width, self.image_height)):
            raise ValueError("image dimensions must be positive integers")
        if self.points is None and self.split == "test":
            return
        if not isinstance(self.points, list):
            raise ValueError("points must be a list (null is reserved for unlabeled test)")
        for index, point in enumerate(self.points):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"point {index} must be an [x, y] pair")
            if any(type(value) not in (int, float) for value in point):
                raise ValueError(f"point {index} must have numeric coordinates")
            x, y = point
            if not (0 <= x < self.image_width and 0 <= y < self.image_height):
                raise ValueError(f"point {index} {point} is non-finite or outside the image")

    @property
    def key(self):
        return self.dataset, self.image_id


def ordered_records(records):
    result = sorted(records, key=lambda record: record.key)
    for previous, current in zip(result, result[1:]):
        if previous.key == current.key:
            raise ValueError(f"duplicate image ID {current.key}")
    return result


def write_manifest(path: Path, records):
    records = ordered_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            for record in records:
                stream.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path, *, splits=None):
    records = []
    excluded = set() if splits is None else {"train", "val", "test"} - set(splits)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
                if excluded and isinstance(item, dict) and item.get("split") in excluded:
                    continue
                record = Record(**item)
                image = Path(record.image_path)
                if not image.is_absolute():
                    image = path.parent / image
                if not image.is_file():
                    raise ValueError(f"missing image {image}")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    return ordered_records(records)
