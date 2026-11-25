import pytest

from loopcrowd.manifest import Record, read_manifest, write_manifest


def test_train_val_scope_does_not_check_official_test_image(tmp_path):
    train_image = tmp_path / "train.jpg"
    train_image.touch()
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [
        Record("1", "fixture", "train", train_image.name, 20, 20, []),
        Record("2", "fixture", "test", "absent-test.jpg", 20, 20, None),
    ])
    assert [r.split for r in read_manifest(path, splits=("train", "val"))] == ["train"]
    with pytest.raises(ValueError, match="absent-test"):
        read_manifest(path)
