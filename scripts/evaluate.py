#!/usr/bin/env python3
"""Evaluate a specific LoopCrowd checkpoint using its saved preprocessing protocol."""

import argparse
import sys
from pathlib import Path

# Keep scripts/profile.py from shadowing Python's profile module via cProfile.
if __name__ == "__main__" and sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch

from loopcrowd.experiment import file_sha, load_checkpoint, make_model, provenance, write_json
from loopcrowd.manifest import read_manifest
from loopcrowd.training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true", help="Confirm protocol and checkpoint are frozen")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--threshold-sweep", action="store_true", help="Evaluate 0.05 to 0.95 on internal validation")
    args = parser.parse_args()
    if args.split == "test" and not args.frozen_test:
        parser.error("official test evaluation requires --frozen-test after model selection")
    if args.threshold_sweep and args.split != "val":
        parser.error("score threshold sweeps are restricted to internal validation")
    device = torch.device(args.device)
    state = load_checkpoint(args.checkpoint, device)
    if file_sha(args.manifest) != state["data_sha256"]:
        parser.error("checkpoint and manifest hashes differ")
    config = state["config"]
    model = make_model(config, initialize=False).to(device)
    model.load_state_dict(state["model"])
    records = [record for record in read_manifest(args.manifest) if record.split == args.split]
    if not records:
        parser.error("no images in the requested split")
    args.output.mkdir(parents=True)
    run = provenance(config, device, args.manifest)
    run["split"] = args.split
    run["checkpoint"] = {"path": str(args.checkpoint.resolve()), "sha256": file_sha(args.checkpoint),
                         "training_commit": state["git_commit"]}
    predictions = [] if args.export_predictions else None
    metrics = evaluate(model, records, args.manifest.parent, config, device, predictions, args.threshold_sweep)
    if predictions is not None:
        write_json(args.output / "predictions.json", predictions)
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "run_manifest.json", run)
    print(metrics)


if __name__ == "__main__":
    main()
