import json
from dataclasses import replace

import pytest

from loopcrowd.datasets import build_manifest
from loopcrowd.manifest import Record, read_manifest, write_manifest


def test_shanghai_nested_mat_and_reproducible_split(shanghai_fixture, tmp_path):
    config, root = shanghai_fixture
    records = build_manifest(config, root)
    assert [record.image_id for record in records] == ["test/IMG_1", "train/IMG_1", "train/IMG_2"]
    assert [record.split for record in records] == ["test", "val", "train"]
    assert records[1].points == []
    assert records[2].points == [[0, 0], [3.9, 3.9], [4, 4], [519.9, 512.9]]
    assert sum(len(record.points) for record in records) == 5
    first, second = tmp_path / "manifest1.jsonl", tmp_path / "manifest2.jsonl"
    write_manifest(first, records)
    write_manifest(second, reversed(records))
    assert first.read_bytes() == second.read_bytes()
    assert read_manifest(first) == records == build_manifest(config, root)


def test_nwpu_released_json_and_split(nwpu_fixture):
    config, root = nwpu_fixture
    records = build_manifest(config, root)
    assert [record.image_id for record in records] == ["0001", "0002", "0003", "3610", "3611"]
    assert [record.split for record in records] == ["train", "train", "val", "test", "test"]
    assert [record.points for record in records] == [[[7.5, 7.5]], [], [[1, 1], [2, 2]], None, None]


@pytest.mark.parametrize("invalid", [
    {"annPoints": [[0, 0]]},
    {"img_id": "0002.jpg", "human_num": 1, "points": [[0, 0]]},
    {"img_id": "0001.jpg", "human_num": 2, "points": [[0, 0]]},
    {"img_id": "0001.jpg", "human_num": 1, "points": [[True, 1]]},
])
def test_nwpu_rejects_unsupported_json(nwpu_fixture, invalid):
    config, root = nwpu_fixture
    (root / "jsons/0001.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="invalid or unsupported NWPU JSON"):
        build_manifest(config, root)


@pytest.mark.parametrize("missing", ["images/0001.jpg", "jsons/0001.json", "images/3610.jpg"])
def test_missing_nwpu_file(nwpu_fixture, missing):
    config, root = nwpu_fixture
    (root / missing).unlink()
    with pytest.raises((OSError, ValueError), match="0001|3610"):
        build_manifest(config, root)


def test_shanghai_missing_pair(shanghai_fixture):
    config, root = shanghai_fixture
    (root / "train_data/images/IMG_1.jpg").unlink()
    with pytest.raises(ValueError, match="missing corresponding image"):
        build_manifest(config, root)


def test_nwpu_duplicate_id_across_splits(nwpu_fixture):
    config, root = nwpu_fixture
    (root / "train.txt").write_text("0003 2 3\n")
    with pytest.raises(ValueError, match="duplicate image ID"):
        build_manifest(config, root)


def test_out_of_bounds_requires_explicit_rejection_log(nwpu_fixture):
    config, root = nwpu_fixture
    annotation = root / "jsons/0001.json"
    data = {"img_id": "0001.jpg", "human_num": 2, "points": [[7.5, 7.5], [8, 0]]}
    annotation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside the image"):
        build_manifest(config, root)
    rejected = []
    assert build_manifest(config, root, rejected)[0].points == [[7.5, 7.5]]
    assert rejected == [{
        "dataset": "nwpu_crowd", "official_split": "train", "image_id": "0001",
        "point_index": 1, "point_xy": [8.0, 0.0],
    }]
    data["points"][1][0] = float("nan")
    annotation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="non-finite"):
        build_manifest(config, root, [])


@pytest.mark.parametrize("points", [
    [[float("nan"), 0]], [[0, float("inf")]], [[-0.01, 0]], [[8, 0]], [[0, 8]], [[0]],
])
def test_invalid_coordinates(points):
    with pytest.raises(ValueError, match="point 0"):
        Record("a", "fixture", "train", "image.jpg", 8, 8, points)


def test_manifest_rejects_duplicate_and_missing_image(nwpu_fixture, tmp_path):
    config, root = nwpu_fixture
    records = build_manifest(config, root)
    path = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="duplicate image ID"):
        write_manifest(path, [records[0], replace(records[0], split="val")])
    write_manifest(path, records)
    (root / "images/0001.jpg").unlink()
    with pytest.raises(ValueError, match="missing image"):
        read_manifest(path)
