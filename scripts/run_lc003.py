#!/usr/bin/env python3
"""Run the frozen LC-003 experiment matrix on one local CUDA device."""

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

from loopcrowd.experiment import file_sha, read_config, write_json
from loopcrowd.manifest import read_manifest
from loopcrowd.model import pretrained_path


MANIFEST_SHA = "e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316"
WEIGHT_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
SCHEDULES = ("constant", "warmup_cosine")
ABLATION = ("cnn_only", "shared_1", "shared_2")
PAIRS = {"shared_2": "unshared_2", "shifted_shared_2": "shifted_unshared_2"}
HISTORICAL = (*ABLATION, "shared_4", "unshared_4", "shifted_shared_2", "shifted_shared_4")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stability(history):
    if len(history) != 40 or history[-1]["global_step"] != 5400:
        raise ValueError("the stability gate requires a completed 40 epoch, 5400 update run")
    late = history[-5:]
    maes = [epoch["validation"]["count"]["mae"] for epoch in late]
    mean = statistics.mean(maes)
    mae_cv = statistics.pstdev(maes) / mean if mean else 0.0
    earlier_loss = statistics.mean(epoch["loss"] for epoch in history[30:35])
    later_loss = statistics.mean(epoch["loss"] for epoch in late)
    loss_delta = (later_loss - earlier_loss) / earlier_loss if earlier_loss else 0.0
    finite = all(math.isfinite(epoch["loss"]) and epoch["gradient_l2"]["nonfinite"] == 0 for epoch in history)
    return {
        "late_mae": mean, "mae_cv": mae_cv, "loss_change": loss_delta, "finite": finite,
        "late_nap": statistics.mean(epoch["validation"]["nap_local_v1"]["nap"] for epoch in late),
        "pass": finite and mae_cv <= 0.10 and abs(loss_delta) <= 0.03,
    }


def choose_schedule(results):
    use_cosine = True
    for variant in ("cnn_only", "shared_2"):
        constant, cosine = [results[schedule][variant] for schedule in SCHEDULES]
        use_cosine &= (
            cosine["pass"] and cosine["late_mae"] < constant["late_mae"]
            and cosine["mae_cv"] < constant["mae_cv"]
        )
    return "warmup_cosine" if use_cosine else "constant"


def run_logged(output, script, arguments):
    log = output.with_suffix(".log")
    print(f"{script}: {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, f"scripts/{script}.py", *map(str, arguments),
                        "--output", str(output)], check=True, stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-train", type=Path, required=True)
    args = parser.parse_args()
    if file_sha(args.manifest) != MANIFEST_SHA or file_sha(pretrained_path()) != WEIGHT_SHA:
        parser.error("local manifest and pretrained file must match the approved cached hashes")
    records = read_manifest(args.manifest)
    for split, count in (("train", 270), ("val", 30)):
        split_records = [record for record in records if record.split == split]
        if len(split_records) != count or any(
            record.points is None or not (args.manifest.parent / record.image_path).is_file()
            for record in split_records
        ):
            parser.error(f"incomplete local {split} files or unexpected split size")
    configs = {name: Path(f"configs/lc003_{name}.json") for name in SCHEDULES}
    constant, cosine = [read_config(configs[name]) for name in SCHEDULES]
    if constant | {"lr_schedule": "warmup_cosine"} != cosine:
        parser.error("the two schedule configs must differ only by lr_schedule")
    for variant in HISTORICAL:
        if not (args.historical_train / variant / "best.pt").is_file():
            parser.error(f"historical checkpoint missing for {variant}")
    args.output.mkdir(parents=True)
    shared_args = ("--manifest", args.manifest, "--device", "cuda")

    def evaluate(model, output, checkpoint):
        run_logged(output, "evaluate", (
            *shared_args, "--checkpoint", model / checkpoint, "--threshold-sweep", "--per-image-counts",
        ))

    def train(schedule, variant, seed=42):
        relative = Path(schedule) / str(seed) / variant
        output = args.output / "train" / relative
        run_logged(output, "train", (*shared_args, "--variant", variant, "--config", configs[schedule], "--seed", seed))
        evaluate(output, args.output / "eval" / relative, "best.pt")
        return stability(load_json(output / "history.json"))

    for shared, independent in PAIRS.items():
        for variant in (shared, independent):
            run_logged(args.output / "profile" / variant, "profile", (
                *shared_args, "--variant", variant, "--config", "configs/lc002_profile.json", "--repeats", 9,
            ))
    for variant in HISTORICAL:
        evaluate(args.historical_train / variant, args.output / "historical_eval" / variant, "best.pt")
    ablation = {schedule: {} for schedule in SCHEDULES}
    for variant in ABLATION:
        for schedule in SCHEDULES:
            ablation[schedule][variant] = train(schedule, variant)
    schedule = choose_schedule(ablation)
    write_json(args.output / "schedule_decision.json", {"schedule": schedule, "ablation": ablation})
    matched = dict(ablation[schedule])
    for variant in ("unshared_2", "shifted_shared_2", "shifted_unshared_2"):
        matched[variant] = train(schedule, variant)
    eligible = [variant for variant, independent in PAIRS.items()
                if matched[variant]["pass"] and matched[independent]["pass"]]
    selected = None
    if matched["cnn_only"]["pass"] and matched["shared_1"]["pass"] and eligible:
        selected = min(eligible, key=lambda variant: (
            matched[variant]["late_mae"], -matched[variant]["late_nap"], tuple(PAIRS).index(variant),
        ))
    decision = {"matched": matched, "eligible_pairs": eligible, "selected_pair": selected,
                "execution": "in_progress" if selected else "not_run"}
    write_json(args.output / "seed_decision.json", decision)
    if selected:
        for seed in (123, 2026):
            for variant in ("cnn_only", "shared_1", selected, PAIRS[selected]):
                train(schedule, variant, seed)
        decision["execution"] = "completed"
        write_json(args.output / "seed_decision.json", decision)
    print("LC-003 queue completed", flush=True)


if __name__ == "__main__":
    main()
