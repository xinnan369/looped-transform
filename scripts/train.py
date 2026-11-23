#!/usr/bin/env python3
"""Train a fixed-loop LoopCrowd variant on its train and internal-val manifest."""

import argparse
import sys
from pathlib import Path

# Keep scripts/profile.py from shadowing Python's profile module via cProfile.
if __name__ == "__main__" and sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch

from loopcrowd.experiment import read_config
from loopcrowd.model import VARIANTS
from loopcrowd.training import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Continue this output directory's last.pt")
    args = parser.parse_args()
    fit(read_config(args.config, args.variant), args.manifest, args.output, torch.device(args.device), args.resume)


if __name__ == "__main__":
    main()
