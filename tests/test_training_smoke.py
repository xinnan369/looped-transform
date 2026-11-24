import json
import os
import subprocess
import sys

from PIL import Image

from loopcrowd.manifest import Record, write_manifest


def test_cpu_train_checkpoint_resume_and_tiled_evaluate(tmp_path):
    manifest = tmp_path / "data.jsonl"
    rows = []
    for index, split in enumerate(("train", "train", "val", "test")):
        path = tmp_path / f"image{index}.png"
        Image.new("RGB", (35, 42), (130, 52, 34)).save(path)
        rows.append(Record(str(index), "fixture", split, path.name, 35, 42, [[0, 0], [8, 9], [9, 9], [10, 10]]))
    write_manifest(manifest, rows)
    config = tmp_path / "config.json"
    properties = {"variant": "shared_2", "tile_size": 32, "tile_overlap": 8}
    config.write_text(json.dumps(properties))
    output, evaluation = tmp_path / "train", tmp_path / "eval"
    environment = os.environ | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}

    def call(*args, success=True):
        process = subprocess.run([sys.executable, *args], env=environment, text=True, capture_output=True)
        assert (process.returncode == 0) == success, process.stdout + process.stderr
        return process

    command = ("scripts/train.py", "--config", str(config), "--manifest", str(manifest),
               "--output", str(output), "--device", "cpu")
    call(*command)
    first = json.loads((output / "run_manifest.json").read_text())
    assert first["completed_epochs"] == 1
    assert first["data"]["training_images"] == 2
    assert first["data"]["validation_images"] == 1
    properties["epochs"] = 2
    config.write_text(json.dumps(properties))
    call(*command, "--resume")
    run = json.loads((output / "run_manifest.json").read_text())
    assert run["completed_epochs"] == 2 and run["resumed_from"]["epoch"] == 1
    assert run["checkpoints"]["last.pt"]["sha256"] != first["checkpoints"]["last.pt"]["sha256"]

    evaluate = ("scripts/evaluate.py", "--checkpoint", str(output / "last.pt"), "--manifest",
                str(manifest), "--output", str(evaluation), "--device", "cpu")
    assert "--frozen-test" in call(*evaluate, "--split", "test", success=False).stderr
    test_sweep = call(*evaluate, "--split", "test", "--frozen-test", "--threshold-sweep", success=False)
    assert "restricted to internal validation" in test_sweep.stderr
    call(*evaluate, "--export-predictions", "--threshold-sweep")
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert metrics["images"] == 1 and metrics["expected_heads"] == 4
    assert metrics["nwpu_official_f1"] == "not_ready"
    assert len(metrics["score_threshold_sweep"]) == 19
    assert metrics["nap_score_floor"] == 0.05
    assert metrics["candidate_diagnostics"]["reconciled_pairs"] == 0
    assert (evaluation / "predictions.json").is_file()
