"""Reconcile existing E36–40 point/count evidence without model inference."""

from collections import defaultdict
from math import isclose, sqrt
from pathlib import Path
from statistics import mean

from loopcrowd.experiment import file_sha
from scripts.prepare_lc007f import history
from scripts.run_lc007a import read_csv

G = Path("reports/LC-007G")
OUT = Path("reports/LC-007H/archived_error_diagnostic.md")
ARMS = ("S4-shift", "U4-shift", "shared_2_aux_025", "unshared_2_aux_025", "S4", "U4")
GROUPS = ("all", "low", "mid", "high")


def archives():
    tables = history()
    rows = read_csv(G / "epoch_index.csv")
    expected = {(s, a, k) for s in (42, 123) for a in ARMS[:2] for k in ("training", "images", "metrics")}
    seen = set()
    for row in rows:
        key = int(row["seed"]), row["arm"], row["kind"]
        path = G / "epochs" / row["seed"] / row["arm"] / f"{row['kind']}.csv"
        if key not in expected or key in seen or path != G / row["path"] or file_sha(path) != row["sha256"]:
            raise ValueError(f"shifted archive index mismatch at {key}")
        tables[key] = read_csv(path)
        if len(tables[key]) != int(row["rows"]):
            raise ValueError(f"shifted archive row count differs at {key}")
        seen.add(key)
    if seen != expected:
        raise ValueError("incomplete shifted archive")
    return tables


def reconcile(tables):
    summaries, annotations = {}, {}
    for seed in (42, 123):
        for arm in ARMS:
            per_image = defaultdict(dict)
            for item in tables[seed, arm, "images"]:
                epoch = int(item["epoch"])
                if epoch < 36 or item["policy"] != "ema":
                    continue
                row = {key: int(item[key]) for key in ("gt", "predicted", "count_error", "abs_count_error",
                                                        "local_tp", "local_fp", "local_fn")}
                gt, pred, signed, absolute, tp, fp, fn = row.values()
                name, group = item["image_id"], item["density_group"]
                if ((gt, pred, signed, absolute) != (tp + fn, tp + fp, fp - fn, abs(fp - fn))
                        or annotations.setdefault(name, (gt, group)) != (gt, group)):
                    raise ValueError(f"inconsistent GT or count evidence at {seed}/{arm}/{name}")
                for selection in ("all", group):
                    scope = epoch, item["stage"], selection
                    if name in per_image[scope]:
                        raise ValueError("duplicate image metric")
                    per_image[scope][name] = row
            for original in tables[seed, arm, "metrics"]:
                epoch = int(original["epoch"])
                if epoch < 36 or original["policy"] != "ema":
                    continue
                stage, group = original["stage"], original["density_group"]
                items = per_image.pop((epoch, stage, group))
                samples = list(items.values())
                n = len(samples)
                values = {"mae": mean(row["abs_count_error"] for row in samples),
                          "rmse": sqrt(mean(row["count_error"] ** 2 for row in samples)),
                          "signed_count_error": sum(row["count_error"] for row in samples),
                          "gt": sum(row["gt"] for row in samples), "images": n,
                          "predicted": sum(row["predicted"] for row in samples)}
                values.update({short: sum(row[f"local_{short}"] for row in samples) for short in ("tp", "fp", "fn")})
                if not all(isclose(value, float(original[k]), abs_tol=1e-12, rel_tol=1e-12) for k, value in values.items()):
                    raise ValueError(f"aggregate does not reconcile at {seed}/{arm}/{epoch}/{stage}/{group}")
                canceled = sum(2 * min(row["local_fp"], row["local_fn"]) for row in samples)
                if canceled + sum(row["abs_count_error"] for row in samples) != values["fp"] + values["fn"]:
                    raise ValueError("count cancellation does not reconcile")
                if set(items) != {name for name, (_, density) in annotations.items() if group == "all" or density == group}:
                    raise ValueError("image coverage differs from the paired reference")
                summaries[seed, arm, epoch, stage, group] = values | {
                    "nap": float(original["local_nap"]), "cancel": canceled,
                    "under": sum(row["count_error"] < 0 for row in samples),
                    "over": sum(row["count_error"] > 0 for row in samples),
                    "exact": sum(row["count_error"] == 0 for row in samples),
                    "signed_mae_bias": values["signed_count_error"] / n}
            if per_image:
                raise ValueError("unreconciled per-image tables")
    return summaries


def write_report(tables, output):
    compact = reconcile(tables)

    def values(arm, *, stage="final", group="all", seed=None):
        chosen = [v for (s, a, _, read, density), v in compact.items()
                  if a == arm and read == stage and density == group and (seed is None or s == seed)]
        if len(chosen) != (5 if seed is not None else 10):
            raise ValueError("requested endpoint lacks a paired horizon")
        return {k: mean(row[k] for row in chosen) for k in chosen[0]}

    def columns(row, keys):
        return " | ".join(f"{row[k]:.6f}" if k == "nap" else f"{row[k]:.4f}" for k in keys)

    keys = ("mae", "rmse", "nap", "signed_mae_bias", "tp", "fp", "fn", "cancel", "under", "over", "exact")
    lines = [
        "# Archived error diagnostic", "",
        f"The CPU audit reconciled {len(compact):,} grouped EMA E36–40 rows against individual image counts. "
        "Each entry weights the archived epochs equally inside each seed and the seeds equally. Count uses strictly "
        "owned score >0.30 and δ=0.50. Local nAP comes from the frozen pooled rank metric and cannot be recomputed from "
        "image count summaries. Positive signed bias means overcount. FP and FN cancel within images according to "
        "FP + FN = |FP−FN| + 2 min(FP,FN).", "",
        "| Arm | MAE | RMSE | Local nAP | Signed error/image | TP | FP | FN | Canceled | Under images | Over images | Exact images |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | {columns(values(arm), keys)} |")
    lines += ["", "## S4-shift minus paired references", "",
              "| Reference | Seed | Density | Δ MAE | Δ local nAP | Δ signed bias/image | Δ TP | Δ FP | Δ FN | Δ canceled |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ref in ARMS[1:4]:
        for seed in (None, 42, 123):
            for group in GROUPS:
                a, b = [values(arm, seed=seed, group=group) for arm in (ARMS[0], ref)]
                delta = {k: a[k] - b[k] for k in a}
                lines.append(f"| {ref} | {seed or 'macro'} | {group} | "
                             f"{columns(delta, ('mae', 'nap', 'signed_mae_bias', 'tp', 'fp', 'fn', 'cancel'))} |")
    lines += ["", "## Four-call stage changes", "",
              "| Arm | Step | Seed | Group | Δ MAE | Δ local nAP | Δ signed bias/image | Δ TP | Δ FP | Δ FN | Δ canceled |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in (*ARMS[:2], *ARMS[-2:]):
        for stage in ("early", "stage3"):
            for seed, group in [(None, group) for group in GROUPS] + [(42, "all"), (123, "all")]:
                before = values(arm, stage=stage, group=group, seed=seed)
                after = values(arm, group=group, seed=seed)
                delta = {k: after[k] - before[k] for k in after}
                lines.append(f"| {arm} | {2 if stage == 'early' else 3}→4 | {seed or 'macro'} | {group} | "
                             f"{columns(delta, ('mae', 'nap', 'signed_mae_bias', 'tp', 'fp', 'fn', 'cancel'))} |")
    lines += ["", "S4-shift records 8.9 more count-side TP and 8.9 fewer FN than U2 at the macro endpoint while its MAE is "
              "1.8075 higher. Per-image signed errors and cancellation explain why the ordering can differ. "
              "Candidate identities, confidence association with unmatched GT or false positives, and same-GT recovery "
              "remain `not_run` / `not_available` in these archived tables. No thresholds, prefixes, epochs or gains were tuned.",
              "", "[Shifted indexes](../LC-007G/epoch_index.csv) · [K=4 indexes](../LC-007E/epoch_index.csv) · "
              "[K=2 indexes](../LC-007B/epoch_index.csv)", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_report(archives(), OUT)
