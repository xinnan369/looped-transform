"""Verify the shifted-window screen and score its frozen geometry comparisons."""

import argparse
from pathlib import Path

from scripts.archive_lc007b import METRICS
from scripts.archive_lc007c import SEEDS, SHARED, UNSHARED
from scripts.archive_lc007e import archive, compact_state_rows
from scripts.prepare_lc007g import SCREEN, history
from scripts.run_lc007a import read_csv, write_csv

JOINT = (("S4-shift", "S4", "primary_shift"), ("U4-shift", "U4", "independent_shift"),
         ("S4-shift", "U4-shift", "matched_sharing"))
CONTRASTS = JOINT + (
    *(("S4-shift", ref, "historical_target") for ref in (SHARED, "cnn_only", "shared_1", UNSHARED)),
    ("S4-shift", "R", "historical_descriptive"),
)
SELECTION = (("S4-shift", ("S4", SHARED, "U4-shift")),)
INTERACTIONS = (("S4-shift", "U4-shift", "S4", "U4", "shift"),)
SHIFT_STEPS = {("stage1", "early"), ("stage3", "final")}
CHANGE_FIELDS = tuple(f"delta_{metric}" for metric in METRICS) + ("delta_canceled_unmatched",)


def phase_identity(row):
    return int(row["epoch"]), row["policy"], row["density_group"], row["before"], row["after"]


def fixed_corrections(tables):
    output = {}
    for seed in SEEDS:
        for arm in ("S4", "U4"):
            metrics = []
            for row in tables[seed, arm, "metrics"]:
                item = row | {"epoch": int(row["epoch"]), **{metric: float(row[metric]) for metric in METRICS}}
                if row["density_group"] == "all":
                    item |= {"nonfinite": int(row["nonfinite"]), **{
                        key: float(row[key]) if row[key] else None for key in ("state_rms", "update_rms", "relative_rms")}}
                metrics.append(item)
            images = [row | {key: int(row[key]) for key in ("epoch", "abs_count_error", "count_error", "local_fp", "local_fn")}
                      for row in tables[seed, arm, "images"]]
            for row in compact_state_rows(metrics, images):
                if (row["before"], row["after"]) in SHIFT_STEPS:
                    output[(seed, arm, *phase_identity(row))] = row
    return output


def paired_geometry(rows, fixed):
    for shifted in rows:
        identity = phase_identity(shifted)
        if identity[-2:] not in SHIFT_STEPS:
            continue
        reference = shifted["arm"].removesuffix("-shift")
        baseline = fixed[(int(shifted["seed"]), reference, *identity)]
        row = {key: shifted[key] for key in ("seed", "arm", "epoch", "policy", "density_group", "before", "after")}
        row["fixed_reference"] = reference
        for key in CHANGE_FIELDS:
            shift, fixed_delta = float(shifted[key]), float(baseline[key])
            row.update({f"shifted_{key}": shift, f"fixed_{key}": fixed_delta, f"did_{key}": shift - fixed_delta})
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tables = history()
    archive(args.run_root, args.output, screen_path=SCREEN, history_tables=lambda: tables,
            contrasts=CONTRASTS, joint=JOINT, strata=INTERACTIONS, selection=SELECTION,
            negative_protocol="retain_K2_and_review_geometry", depth_calls=4)
    source = args.output / "stagewise_correction.csv"
    if source.is_file():
        write_csv(args.output / "paired_shift_steps.csv", list(paired_geometry(read_csv(source), fixed_corrections(tables))))


if __name__ == "__main__":
    main()
