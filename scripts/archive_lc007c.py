"""Recompute stagewise correction from the frozen LC-007B CSV evidence on CPU."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean

from scripts.run_lc007a import read_csv, write_csv

ARCHIVE = Path("reports/LC-007B")
OUTPUT = Path("reports/LC-007C")
PARENT = "0aeaea0562f075fd28ecaa48c6400193a7cf22f4"
FREEZE = "6471f222dbe1480ff7fca1ef0291215f2498d32f"
INDEX_SHA = "6387f93632e3a5c5f3fcbba605697bc3a889c039469cdc29e962923be36e79a4"
PREFLIGHT_SHA = "d2c01019d3fefb60d9712bb38c4f39f3ef9123defa0a7df276ae96d97c68a042"
SEEDS = (42, 123)
POLICIES = ("live", "ema")
GROUPS = ("all", "low", "mid", "high")
GROUP_COUNTS = {"all": (40, 4295), "low": (13, 566), "mid": (16, 1511), "high": (11, 2218)}
EPOCHS = tuple(range(36, 41))
SHARED, UNSHARED = "shared_2_aux_025", "unshared_2_aux_025"
ARMS = {SHARED: ("early", "final"), UNSHARED: ("early", "final"),
        "shared_1": ("final",), "cnn_only": ("final",)}
METRICS = ("mae", "rmse", "local_nap", "tp", "fp", "fn", "signed_count_error", "predicted")


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def read_frozen_files(directory):
    if (checksum(directory / "epoch_index.csv") != INDEX_SHA or
            checksum(directory / "preflight.json") != PREFLIGHT_SHA):
        raise ValueError("LC-007B parent archive index or preflight SHA-256 differs")
    index = read_csv(directory / "epoch_index.csv")
    expected = {(seed, arm, kind) for seed in SEEDS for arm in ARMS for kind in ("training", "metrics", "images")}
    tables = {}
    for row in index:
        key = int(row["seed"]), row["arm"], row["kind"]
        source = Path(row["path"])
        expected_path = Path("epochs") / str(key[0]) / key[1] / f"{key[2]}.csv"
        if key not in expected or key in tables or source != expected_path or row["status"] != "completed":
            raise ValueError(f"unexpected LC-007B index entry: {key}")
        path = directory / source
        if path.stat().st_size != int(row["bytes"]) or checksum(path) != row["sha256"]:
            raise ValueError(f"frozen indexed bytes or SHA-256 differ: {path}")
        rows = read_csv(path)
        if len(rows) != int(row["rows"]):
            raise ValueError(f"frozen row count differs: {path}")
        tables[key] = rows
    if tables.keys() != expected:
        raise ValueError("the LC-007B index is incomplete")
    return tables


def check_training(training):
    for epoch, row in enumerate(training, 1):
        if (int(row["epoch"]) != epoch or int(row["ema_step"]) != 180 * epoch
                or int(row["global_step"]) != 180 * epoch or int(row["optimizer_updates"]) != 180
                or int(row["amp_skipped_updates"]) or int(row["gradient_l2_nonfinite"])):
            raise ValueError("LC-007B training trajectory is incomplete or nonfinite")
    if len(training) != 40:
        raise ValueError("LC-007B checkpoint horizon differs")


def load_readouts(tables, directory):
    ids = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))["ordered_val_ids"]
    expected_ids = set(ids)
    if len(ids) != 40 or len(expected_ids) != len(ids):
        raise ValueError("frozen validation IDs differ")
    packed, image_labels = {}, {}
    for seed in SEEDS:
        for arm, stages in ARMS.items():
            check_training(tables[seed, arm, "training"])
            grouped, metrics = defaultdict(dict), {}
            for row in tables[seed, arm, "images"]:
                key = int(row["epoch"]), row["policy"], row["stage"]
                if key[0] not in EPOCHS:
                    continue
                name, density = row["image_id"], row["density_group"]
                values = {field: int(row[field]) for field in
                          ("gt", "predicted", "count_error", "abs_count_error", "local_tp", "local_fp", "local_fn")}
                gt, predicted, signed, absolute, tp, fp, fn = values.values()
                if (name in grouped[key] or density not in GROUPS[1:] or min(gt, predicted, tp, fp, fn) < 0
                        or (gt, predicted, signed, absolute) != (tp + fn, tp + fp, fp - fn, abs(fp - fn))):
                    raise ValueError(f"inconsistent count/matching evidence: {seed}/{arm}/{key}/{name}")
                label = gt, density
                if image_labels.setdefault(name, label) != label:
                    raise ValueError(f"image label or density group differs: {name}")
                grouped[key][name] = values | {"density_group": density}
            for row in tables[seed, arm, "metrics"]:
                key = int(row["epoch"]), row["policy"], row["stage"], row["density_group"]
                if key[0] not in EPOCHS:
                    continue
                if key in metrics:
                    raise ValueError(f"duplicate grouped metric for {seed}/{arm}/{key}")
                metrics[key] = {field: float(row[field]) for field in METRICS} | {
                    "images": int(row["images"]), "gt": int(row["gt"])}
            expected = {(e, p, s) for e in EPOCHS for p in POLICIES for s in stages}
            if grouped.keys() != expected or metrics.keys() != {(*key, group) for key in expected for group in GROUPS}:
                raise ValueError(f"missing or extra grouped E36–40 scopes: {seed}/{arm}")
            for key, images in grouped.items():
                if images.keys() != expected_ids:
                    raise ValueError(f"image ID membership differs: {seed}/{arm}/{key}")
                for group in GROUPS:
                    items = [images[name] for name in ids if group == "all" or images[name]["density_group"] == group]
                    reported = metrics[(*key, group)]
                    check_aggregate(items, reported, group)
                    packed[(seed, arm, *key, group)] = reported, {
                        name: images[name] for name in ids if group == "all" or images[name]["density_group"] == group}
    return packed


def check_aggregate(images, saved, label):
    sums = {"gt": sum(r["gt"] for r in images), "images": len(images)}
    sums.update({field: sum(r["local_" + field] for r in images) for field in ("tp", "fp", "fn")})
    sums.update({output: sum(row[field] for row in images)
                 for output, field in (("predicted", "predicted"), ("signed_count_error", "count_error"))})
    if (sums["images"], sums["gt"]) != GROUP_COUNTS[label]:
        raise ValueError(f"frozen density group population differs: {label}")
    sums["mae"] = mean(r["abs_count_error"] for r in images)
    sums["rmse"] = sqrt(mean(r["count_error"] ** 2 for r in images))
    if any(not isclose(value, saved[key], abs_tol=1e-12, rel_tol=1e-12) for key, value in sums.items()):
        raise ValueError(f"archived image/group aggregates differ: {label}")
    if not isfinite(saved["local_nap"]) or not 0 <= saved["local_nap"] <= 1:
        raise ValueError(f"frozen pooled nAP undefined or invalid: {label}")


def correction(early, final):
    """Count error changes are over aggregate counts, without GT-identity claims."""
    before, after = early["abs_count_error"], final["abs_count_error"]
    fp, fn = [final[key] - early[key] for key in ("local_fp", "local_fn")]
    canceled = 2 * (min(final["local_fp"], final["local_fn"]) - min(early["local_fp"], early["local_fn"]))
    if after - before != fp + fn - canceled:
        raise ValueError("count cancellation identity failed")
    return {"early_abs_error": before, "final_abs_error": after, "correction_abs_error": after - before,
            "early_signed_error": early["count_error"], "final_signed_error": final["count_error"],
            "delta_fp": fp, "delta_fn": fn, "delta_canceled_unmatched": canceled}


def compare(readouts):
    grouped, image_rows = [], []
    for seed in SEEDS:
        for policy in POLICIES:
            for epoch in EPOCHS:
                for group in GROUPS:
                    meta = {"seed": seed, "epoch": epoch, "policy": policy, "density_group": group}
                    stages = {arm: {stage: readouts[seed, arm, epoch, policy, stage, group]
                                    for stage in names} for arm, names in ARMS.items()}
                    row = meta | {"images": GROUP_COUNTS[group][0]}
                    shared = stages[SHARED]
                    independent = stages[UNSHARED]
                    for metric in METRICS:
                        se, sf = [shared[stage][0][metric] for stage in ("early", "final")]
                        ue, uf = [independent[stage][0][metric] for stage in ("early", "final")]
                        before, after = se - ue, sf - uf
                        row.update({
                            f"shared_early_{metric}": se, f"shared_final_{metric}": sf,
                            f"unshared_early_{metric}": ue, f"unshared_final_{metric}": uf,
                            f"shared_correction_{metric}": sf - se, f"unshared_correction_{metric}": uf - ue,
                            f"early_gap_{metric}": before, f"final_gap_{metric}": after,
                            f"correction_did_{metric}": (sf - se) - (uf - ue),
                        })
                        for reference in ("shared_1", "cnn_only"):
                            ref = stages[reference]["final"][0][metric]
                            row[f"{reference}_final_{metric}"] = ref
                            row[f"shared_final_minus_{reference}_{metric}"] = sf - ref
                    names = shared["early"][1].keys()
                    paired = {arm: [correction(outputs["early"][1][name], outputs["final"][1][name])
                                    for name in names] for arm, outputs in (("shared", shared), ("unshared", independent))}
                    for arm, items in paired.items():
                        for label, count in (("improved", sum(r["correction_abs_error"] < 0 for r in items)),
                                             ("tied", sum(r["correction_abs_error"] == 0 for r in items)),
                                             ("worsened", sum(r["correction_abs_error"] > 0 for r in items))):
                            row[f"{arm}_{label}_images"] = count
                            row[f"{arm}_{label}_fraction"] = count / len(items)
                        row[f"{arm}_delta_canceled_unmatched"] = sum(r["delta_canceled_unmatched"] for r in items)
                        image_mae_change = mean(r["correction_abs_error"] for r in items)
                        if not isclose(image_mae_change, row[f"{arm}_correction_mae"], abs_tol=1e-12):
                            raise ValueError("image MAE change differs from stagewise MAE change")
                    row["did_delta_canceled_unmatched"] = (
                        row["shared_delta_canceled_unmatched"] - row["unshared_delta_canceled_unmatched"])
                    if group == "all":
                        for name, s, u in zip(names, paired["shared"], paired["unshared"]):
                            before = s["early_abs_error"] - u["early_abs_error"]
                            after = s["final_abs_error"] - u["final_abs_error"]
                            image_rows.append(meta | {"density_group": shared["early"][1][name]["density_group"],
                                                       "image_id": name, "gt": shared["early"][1][name]["gt"]}
                                              | {f"shared_{k}": v for k, v in s.items()}
                                              | {f"unshared_{k}": v for k, v in u.items()}
                                              | {"early_abs_error_gap": before, "final_abs_error_gap": after,
                                                 "abs_error_correction_did": after - before})
                    grouped.append(row)
    return grouped, image_rows


def summaries(rows):
    """Macro average epochs inside seed before averaging the two seeds."""
    result = []
    for label, epochs in (("late", EPOCHS), ("e40", (40,))):
        per_seed = []
        for seed in SEEDS:
            for policy in POLICIES:
                for group in GROUPS:
                    subset = [row for row in rows if (row["seed"], row["policy"], row["density_group"])
                              == (seed, policy, group) and row["epoch"] in epochs]
                    values = numeric_average(subset)
                    per_seed.append({"seed": seed, "epoch": label, "policy": policy, "density_group": group,
                                     "seed_count": 1, "epochs_per_seed": len(epochs)} | values)
        result.extend(per_seed)
        for policy in POLICIES:
            for group in GROUPS:
                subset = [row for row in per_seed if (row["policy"], row["density_group"]) == (policy, group)]
                result.append({"seed": "macro", "epoch": label, "policy": policy, "density_group": group,
                               "seed_count": 2, "epochs_per_seed": len(epochs)} | numeric_average(subset))
    return result


def numeric_average(rows):
    skip = {"seed", "epoch", "seed_count", "epochs_per_seed", "policy", "density_group"}
    return {key: mean(row[key] for row in rows) for key in rows[0] if key not in skip}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    tables = read_frozen_files(args.archive)
    grouped, paired = compare(load_readouts(tables, args.archive))
    summary = summaries(grouped)
    write_csv(args.output / "archive_correction.csv", grouped + summary)
    write_csv(args.output / "archive_image_correction.csv", paired)
    provenance = {
        "source_commit": PARENT, "source_freeze_commit": FREEZE,
        "archive_index_sha256": checksum(args.archive / "epoch_index.csv"),
        "preflight_sha256": checksum(args.archive / "preflight.json"),
        "verified_index_entries": len(tables), "indexed_rows": sum(map(len, tables.values())),
        "late_grouped_rows": len(grouped), "summary_rows": len(summary), "paired_image_rows": len(paired),
        "seeds": SEEDS, "policies": POLICIES, "groups": GROUP_COUNTS, "epochs": EPOCHS,
        "archive_correction_sha256": checksum(args.output / "archive_correction.csv"),
        "archive_image_correction_sha256": checksum(args.output / "archive_image_correction.csv"),
    }
    (args.output / "archive_integrity.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
