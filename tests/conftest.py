import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat


@pytest.fixture
def shanghai_fixture(tmp_path):
    root = tmp_path / "shanghai"
    for split, images in {
        "train_data": {
            "IMG_2": [[0, 0], [3.9, 3.9], [4, 4], [519.9, 512.9]],
            "IMG_1": [],
        },
        "test_data": {"IMG_1": [[3, 4]]},
    }.items():
        base = root / split
        (base / "images").mkdir(parents=True)
        (base / "ground_truth").mkdir()
        for name, points in images.items():
            Image.new("RGB", (520, 513)).save(base / "images" / f"{name}.jpg")
            structure = np.empty((1, 1), dtype=[("location", "O"), ("number", "O")])
            structure["location"][0, 0] = np.asarray(points).reshape(-1, 2)
            structure["number"][0, 0] = np.array([[len(points)]])
            cell = np.empty((1, 1), dtype=object)
            cell[0, 0] = structure
            savemat(base / "ground_truth" / f"GT_{name}.mat", {"image_info": cell})

    config = json.loads(Path("configs/shanghaitech_a.json").read_text())
    config["validation"]["fraction"] = 0.5
    path = tmp_path / "shanghai.json"
    path.write_text(json.dumps(config))
    return path, root


@pytest.fixture
def nwpu_fixture(tmp_path):
    root = tmp_path / "nwpu"
    (root / "images").mkdir(parents=True)
    (root / "jsons").mkdir()
    (root / "train.txt").write_text("0002 1 0\n0001 0 1\n")
    (root / "val.txt").write_text("0003 2 3\n")
    (root / "test.txt").write_text("3610 1 3\n3611\n")
    for image_id, points in {
        "0001": [[7.5, 7.5]],
        "0002": [],
        "0003": [[1, 1], [2, 2]],
        "3610": None,
        "3611": None,
    }.items():
        Image.new("RGB", (8, 8)).save(root / "images" / f"{image_id}.jpg")
        if points is not None:
            payload = {"img_id": f"{image_id}.jpg", "human_num": len(points), "points": points}
            (root / "jsons" / f"{image_id}.json").write_text(json.dumps(payload))
    return Path("configs/nwpu_crowd.json"), root
