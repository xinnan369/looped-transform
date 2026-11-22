#!/usr/bin/env python3
"""Prepare the pinned P2PNet SHHA loader's two lists from a ShanghaiTech manifest."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from loopcrowd.manifest import read_manifest


def prepare(manifest, output, evaluation_split):
    records = read_manifest(manifest)
    datasets = {record.dataset for record in records}
    if len(datasets) != 1 or not datasets <= {"shanghaitech_a", "shanghaitech_b"}:
        raise ValueError("expected one ShanghaiTech Part A or Part B manifest")
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new directory: {output}")
    selected = {
        "train": [record for record in records if record.split == "train"],
        "test": [record for record in records if record.split == evaluation_split],
    }
    if any(not rows for rows in selected.values()):
        raise ValueError("manifest must have nonempty train and requested evaluation splits")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        for name, rows in selected.items():
            images, points = stage / name / "images", stage / name / "points"
            images.mkdir(parents=True)
            points.mkdir()
            lines = []
            for index, record in enumerate(rows, 1):
                source = (manifest.resolve().parent / record.image_path).resolve()
                # The upstream loader extracts its numeric ID from the image basename.
                image = images / f"IMG_{index}.jpg"
                label = points / f"IMG_{index}.txt"
                image.symlink_to(source)
                label.write_text(
                    "".join(f"{x} {y}\n" for x, y in record.points), encoding="utf-8"
                )
                lines.append(f"{image.relative_to(stage).as_posix()} {label.relative_to(stage).as_posix()}\n")
            (stage / f"shanghai_tech_part_a_{name}.list").write_text("".join(lines), encoding="utf-8")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New derived-data directory")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    try:
        prepare(args.manifest, args.output, args.evaluation_split)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Wrote P2PNet lists to {args.output}. Evaluation split: {args.evaluation_split}.")


if __name__ == "__main__":
    main()
