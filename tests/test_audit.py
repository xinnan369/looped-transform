import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from loopcrowd.audit import make_audit
from loopcrowd.manifest import Record, read_manifest


def record(name, points, width=9, height=9, split="train"):
    return Record(name, "fixture", split, "unused.jpg", width, height, points)


def test_grid_occupancy_and_point_weighted_overflow():
    dense = record("dense", [[0, 0]] * 10 + [[4, 4], [8.999, 8.999]])
    sparse = record("sparse", [[7, 7]])
    empty = record("empty", [])
    report = make_audit([sparse, empty, dense], "synthetic", ["full-image"], 42)
    group = report["groups"][0]
    assert group["head_count"] == 13
    assert group["heads_per_image"]["p50"] == 1
    assert group["nearest_neighbor_px"]["p50"] == 0
    for stride, cells, expected in (
        ("4", 9, [9, 8, 6, 2]),
        ("8", 4, [10, 9, 7, 3]),
    ):
        grid = group["grids"][stride]
        assert grid["total_cells"] == 3 * cells
        assert grid["occupied_cells"] == (4 if stride == "4" else 3)
        assert grid["occupied_fraction"] == grid["occupied_cells"] / (3 * cells)
        dense_grid = group["images"][0]["grids"][stride]
        for k, lost in zip(("1", "2", "4", "8"), expected):
            assert grid["overflow"][k] == {"lost_points": lost, "point_loss_rate": lost / 13}
            assert dense_grid["overflow"][k]["point_loss_rate"] == lost / 12
    empty_row = group["images"][1]
    assert empty_row["nearest_neighbor_px"] is None
    assert empty_row["grids"]["4"]["cell_histogram"] == {"0": 9}
    assert empty_row["grids"]["4"]["overflow"]["1"]["point_loss_rate"] is None


def test_nearest_neighbor_is_local_and_euclidean():
    images = [record("a", [[0, 0], [3, 4]]), record("b", [[0, 0]])]
    group = make_audit(images, "synthetic", ["full-image"], 42)["groups"][0]
    assert set(group["nearest_neighbor_px"].values()) == {5.0}
    assert group["images"][1]["nearest_neighbor_px"] is None


def test_extreme_single_cell_density():
    size = 20_000
    group = make_audit([record("dense", [[1, 1]] * size)], "synthetic", ["full-image"], 42)["groups"][0]
    for grid in group["grids"].values():
        assert grid["max_points_per_cell"] == size
        assert grid["overflow"]["8"]["lost_points"] == size - 8


def test_crop_half_open_boundaries_and_padding():
    source = record("wide", [[x, 0] for x in range(514)], 514, 1)
    report = make_audit([source], "synthetic", ["full-image", "crop-512"], 42)
    full, cropped = report["groups"]
    assert full["head_count"] == 514
    assert cropped["source_head_count"] == 514
    assert cropped["head_count"] == 512
    image = cropped["images"][0]
    assert image["crop_origin_xy"][0] in [0, 1, 2]
    assert image["crop_origin_xy"][1] == 0
    assert (image["width"], image["height"]) == (512, 512)
    assert image["grids"]["4"]["total_cells"] == 128**2
    assert image["grids"]["4"]["occupied_cells"] == 128
    assert report == make_audit([source], "synthetic", ["full-image", "crop-512"], 42)


def test_aggregate_and_unlabeled_test_are_distinct():
    records = [record("a", [[0, 0]]), record("b", [], split="val"), record("c", None, split="test")]
    report = make_audit(records, "synthetic", ["full-image"], 42, include_images=False)
    assert [(g["split"], g["image_count"], g["head_count"]) for g in report["groups"]] == [
        ("all", 2, 1), ("train", 1, 1), ("val", 1, 0),
    ]
    assert all("images" not in group for group in report["groups"])
    assert report["unlabeled"] == [{"dataset": "fixture", "split": "test", "image_count": 1}]
    with pytest.raises(ValueError, match="no labeled images"):
        make_audit(records, "synthetic", ["full-image"], 42, "test")


@pytest.mark.parametrize("fixture_name,points", [("shanghai_fixture", 5), ("nwpu_fixture", 3)])
def test_cpu_cli_end_to_end(fixture_name, points, request, tmp_path):
    config, root = request.getfixturevalue(fixture_name)
    manifest = tmp_path / "derived" / "manifest.jsonl"
    prefix = tmp_path / "report" / "synthetic"
    subprocess.run(
        [sys.executable, "scripts/build_manifest.py", "--config", str(config),
         "--root", str(root), "--output", str(manifest)],
        check=True, capture_output=True,
    )
    assert sum(len(record.points or []) for record in read_manifest(manifest)) == points
    outputs = []
    for _ in range(2):
        subprocess.run(
            [sys.executable, "scripts/audit_dataset.py", "--manifest", str(manifest),
             "--output-prefix", str(prefix), "--source", "synthetic"],
            check=True, capture_output=True,
        )
        outputs.append(prefix.with_suffix(".json").read_bytes())
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0])
    schema = json.loads(Path("reports/LC-001/data_audit.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    assert report["source"] == "synthetic"
    assert {g["mode"] for g in report["groups"]} == {"full-image", "crop-512"}
    assert "synthetic" in prefix.with_suffix(".md").read_text(encoding="utf-8")
