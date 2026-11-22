#!/usr/bin/env python3
"""Audit labeled points on full images and/or deterministic 512×512 crops."""

import argparse
import json
from pathlib import Path

from loopcrowd.audit import make_audit, to_markdown
from loopcrowd.manifest import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path, help="Writes .json and .md")
    parser.add_argument("--source", required=True, choices=("real", "synthetic"))
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--mode", default="both", choices=("full-image", "crop-512", "both"))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--aggregate-only", action="store_true", help="Omit image IDs and per-image rows")
    args = parser.parse_args()
    modes = ("full-image", "crop-512") if args.mode == "both" else (args.mode,)
    try:
        report = make_audit(
            read_manifest(args.manifest), args.source, modes, args.seed, args.split,
            include_images=not args.aggregate_only,
        )
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix, content in (
            (".json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
            (".md", to_markdown(report)),
        ):
            Path(f"{args.output_prefix}{suffix}").write_text(content, encoding="utf-8", newline="\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Wrote {args.output_prefix}.json and {args.output_prefix}.md")


if __name__ == "__main__":
    main()
