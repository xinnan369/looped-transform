#!/usr/bin/env python3
"""Convert one dataset release into a validated JSONL manifest."""

import argparse
import json
from pathlib import Path

from loopcrowd.datasets import build_manifest
from loopcrowd.manifest import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path, help="Original dataset directory")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--out-of-bounds-rejections", type=Path,
        help="Exclude finite out-of-image points and record each exclusion in this JSON file",
    )
    args = parser.parse_args()
    rejections_path = args.out_of_bounds_rejections
    rejections = [] if rejections_path is not None else None
    try:
        if rejections_path is not None and rejections_path.resolve() == args.output.resolve():
            raise ValueError("the rejection log and manifest must have different paths")
        records = build_manifest(args.config, args.root, rejections)
        if rejections_path is not None:
            rejections_path.parent.mkdir(parents=True, exist_ok=True)
            rejections_path.write_text(
                json.dumps(rejections, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
        write_manifest(args.output, records)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    labeled_points = sum(len(record.points) for record in records if record.points is not None)
    print(f"Wrote {len(records)} images and {labeled_points} labeled points")
    if rejections is not None:
        print(f"Excluded {len(rejections)} out-of-image points. See {rejections_path}")


if __name__ == "__main__":
    main()
