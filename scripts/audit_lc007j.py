"""Reconcile fixed predictions from the two archived seed-42 S4-shift runs."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from math import isclose, sqrt
from pathlib import Path
from statistics import mean

G = Path("reports/LC-007G")
I = Path("reports/LC-007I/training_results")
GROUPS = ("all", "low", "mid", "high")
POLICIES = ("live", "ema")
STAGES = ("stage1", "early", "stage3", "final")
KINDS = ("training", "images", "metrics")
FIELDS = ("gt", "predicted", "count_error", "abs_count_error", "local_tp", "local_fp", "local_fn")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(f.read()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_index(folder, expected_digest, selector):
    index = folder / "epoch_index.csv"
    if digest(index) != expected_digest:
        raise ValueError("source index digest differs")
    tables = {}
    for item in read_csv(index):
        if not selector(item):
            continue
        kind, path = item["kind"], folder / item["path"]
        if kind not in KINDS or kind in tables or path.stat().st_size != int(item["bytes"]) or digest(path) != item["sha256"]:
            raise ValueError(f"indexed bytes differ: {path}")
        tables[kind] = read_csv(path)
        if len(tables[kind]) != int(item["rows"]):
            raise ValueError(f"indexed row count differs: {path}")
    if set(tables) != set(KINDS):
        raise ValueError("source index is incomplete")
    return tables


def record(entry):
    x = {key: int(entry[key]) for key in FIELDS}
    gt, pred, signed, absolute, tp, fp, fn = (x[key] for key in FIELDS)
    canceled = 2 * min(fp, fn)
    if (min(gt, pred, tp, fp, fn) < 0 or gt != tp + fn or pred != tp + fp or signed != fp - fn
            or absolute != abs(signed) or fp + fn != absolute + canceled
            or ("canceled_unmatched" in entry and int(entry["canceled_unmatched"]) != canceled)):
        raise ValueError("image count or cancellation identity differs")
    return x | {"image_id": entry["image_id"], "density_group": entry["density_group"], "canceled_unmatched": canceled}


def pack(entries, expected_ids, labels):
    by = defaultdict(dict)
    for entry in entries:
        scope = int(entry["epoch"]), entry["policy"], entry["stage"]
        x = record(entry)
        image = x["image_id"]
        if image in by[scope] or labels.setdefault(image, (x["gt"], x["density_group"])) != (x["gt"], x["density_group"]):
            raise ValueError(f"duplicate or conflicting image in {scope}")
        if x["density_group"] not in GROUPS[1:]:
            raise ValueError("unknown density group")
        by[scope][image] = x
    if set(by) != {(e, p, s) for e in range(1, 41) for p in POLICIES for s in STAGES}:
        raise ValueError("epoch, policy or stage coverage differs")
    if any(list(group) != expected_ids for group in by.values()):
        raise ValueError("ordered validation IDs differ")
    return by


def ratio(num, den):
    return num / den if den else None


def summary(items):
    n = len(items)
    if not n:
        raise ValueError("empty known density group")
    sums = {key: sum(r[key] for r in items) for key in (*FIELDS, "canceled_unmatched")}
    errors = [x["count_error"] for x in items]
    under, over = [x for x in errors if x < 0], [x for x in errors if x > 0]
    return dict(images=n, gt=sums["gt"], predicted=sums["predicted"], signed_count_error=sums["count_error"],
                mae=sums["abs_count_error"] / n, rmse=sqrt(sum(e * e for e in errors) / n),
                tp=sums["local_tp"], fp=sums["local_fp"], fn=sums["local_fn"], canceled=sums["canceled_unmatched"],
                precision=ratio(sums["local_tp"], sums["predicted"]), recall=ratio(sums["local_tp"], sums["gt"]),
                under=len(under), exact=n-len(under)-len(over), over=len(over),
                under_magnitude=-sum(under), over_magnitude=sum(over))


def archive_metrics(entries):
    result = {(int(r["epoch"]), r["policy"], r["stage"], r["density_group"]): r for r in entries}
    if len(result) != len(entries):
        raise ValueError("duplicate grouped metrics")
    return result


def reconcile(by, metrics):
    result = {}
    for scope, images in by.items():
        for group in GROUPS:
            x = summary([i for i in images.values() if group == "all" or i["density_group"] == group])
            key = (*scope, group)
            checked = metrics.pop(key)
            for n in ("images", "gt", "predicted", "signed_count_error", "tp", "fp", "fn", "mae", "rmse"):
                if not isclose(x[n], float(checked[n]), abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError(f"archive group differs at {key}/{n}")
            result[key] = x | {"archived_nap": float(checked["local_nap"])}
    if metrics:
        raise ValueError("unreconciled archive groups")
    return result


def paired(before, after):
    keys = ("predicted", "count_error", "abs_count_error", "local_tp", "local_fp", "local_fn", "canceled_unmatched")
    changes = {k: after[k] - before[k] for k in keys}
    if changes["count_error"] != changes["local_fp"] - changes["local_fn"]:
        raise ValueError("paired signed error decomposition differs")
    return {f"{label}_{k}": row[k] for label, row in (("512", before), ("768", after), ("delta", changes)) for k in keys}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        out = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def average(rows):
    return {k: mean(vals) if len(vals) == len(rows) else None for k in rows[0]
            for vals in ([r[k] for r in rows if r[k] is not None],)}


def local_source(record, image_path):
    file = Path(record["path"])
    if not file.is_file():
        return {"checkpoint": "not_available", "e40_candidate_identities": "not_available"}
    if file.stat().st_size != record["bytes"] or digest(file) != record["sha256"]:
        raise ValueError("local retained checkpoint digest differs")
    raw = read_json(image_path) if image_path.is_file() else []
    if any(k in row for row in raw for k in ("candidate_ids", "gt_ids", "predictions")):
        raise ValueError("raw identity source exists and requires review")
    names = {"candidates", "candidate_ids", "predictions", "gt_matches", "identity"}
    possible = sorted(str(p) for p in file.parent.rglob("*") if p.is_file() and any(k in p.stem.lower() for k in names))
    return {"checkpoint": "sha256_verified", "path": str(file), "sha256": record["sha256"],
            "local_e40_image_rows": len(raw), "identity_named_files": possible,
            "e40_candidate_identities": "not_available" if not possible else "requires_inspection"}


def previous_512_export(checkpoint, ids):
    source = Path("reports/LC-007H/e40_execution_evidence.json")
    if digest(source) != "726c6b0c1a7b7626888214730ceb60f09fcdeb7a40083d1dd4f411fdac20c352":
        raise ValueError("prior E40 diagnostic index differs")
    prior = read_json(source)["cases"]["42/S4-shift"]
    ref = prior["images"]
    path = Path(ref["path"])
    if not path.is_file():
        return {"identities": "not_available_locally", "indexed_source": ref}
    if (path.stat().st_size != ref["bytes"] or digest(path) != ref["sha256"]
            or prior["checkpoint"]["sha256"] != checkpoint["sha256"] or prior["parity"] != "pass"):
        raise ValueError("prior 512 identity evidence differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [row["image_id"] for row in rows] != ids or any(
        "midpoint_candidate_ids" not in row["midpoint"] or "stage3_to_final" not in row for row in rows
    ):
        raise ValueError("prior 512 identity export layout differs")
    return {"identities": "midpoint_export_and_gt_transitions_only", "rows": len(rows), "indexed_source": ref}


def audit(output, *, inspect_local=False):
    tables = {
        "512": verify_index(G, "c68d6328cd7da766a8b124793c3852a4c524b94413073d430b9e4ddf9f36bf38",
                            lambda row: (row["seed"], row["arm"]) == ("42", "S4-shift")),
        "768": verify_index(I, "c58e21aeaf1d368a1d72e822cdea72f42ad3d3aed6cfaccc9d903a0dc80aeb0d", lambda row: True),
    }
    checkpoints = {}
    for name, directory, want in (
        ("512", G, "b09216ca1f028529e255f732a9efb96f8319c5265e3424dab9a05f83cf23eea3"),
        ("768", I, "ae11e234de25e6f35fc7a850fca4c82f533637f26f3d49c13d66a8d6fa4c6c44"),
    ):
        p = directory / "run_evidence.json"
        if digest(p) != want:
            raise ValueError("historical run evidence differs")
        x = read_json(p)
        case = x["case"] if name == "768" else next(c for c in x["cases"] if c["seed"] == 42 and c["arm"] == "S4-shift")
        if case["status"] != "completed" or case["epochs"] != 40 or case["successful_updates"] != 7200:
            raise ValueError("historical terminal checkpoint differs")
        record_ref = case["retained_checkpoint"]
        if tables[name]["training"][-1]["checkpoint_sha256"] != record_ref["sha256"]:
            raise ValueError("indexed E40 checkpoint differs from execution evidence")
        checkpoints[name] = {"source_commit": x["execution"]["code_commit"], "evidence_sha256": want,
                             "checkpoint": record_ref}
        if inspect_local:
            checkpoints[name]["local"] = local_source(record_ref, Path(record_ref["path"]).parent / "images/40.json")
    init = G / "initialization_audit.json"
    if digest(init) != "52b8170975a9d1dbb04878f0682d4ac2f27d0bbcdf0f75756ddf82027f323efb":
        raise ValueError("historical validation ordering digest differs")
    ids = read_json(init)["data"]["ordered_val_ids"]
    if inspect_local:
        checkpoints["512"]["prior_read_only_export"] = previous_512_export(checkpoints["512"]["checkpoint"], ids)
    labels = {}
    cube = {name: pack(t["images"], ids, labels) for name, t in tables.items()}
    groups = {name: reconcile(by, archive_metrics(tables[name]["metrics"])) for name, by in cube.items()}
    if len(labels) != 40 or sum(gt for gt, _ in labels.values()) != 4295:
        raise ValueError("frozen labels differ")
    primary_rows, grouped = [], []
    for key in sorted(cube["512"]):
        e, p, s = key
        if e >= 36:
            for image, x in cube["512"][key].items():
                primary_rows.append({"epoch": e, "policy": p, "stage": s, "image_id": image,
                                     "density_group": x["density_group"], "gt": x["gt"],
                                     **paired(x, cube["768"][key][image])})
        for name, stats in groups.items():
            for g in GROUPS:
                grouped.append({"epoch": e, "policy": p, "stage": s, "density_group": g, "crop": name, **stats[(*key, g)]})
    final = {name: average([stats[e, "ema", "final", "all"] for e in range(36, 41)]) for name, stats in groups.items()}
    epochs = [r for r in primary_rows if (r["policy"], r["stage"]) == ("ema", "final")]
    occurrence = {i: mean(r["delta_local_fp"] for r in epochs if r["image_id"] == i) for i in ids}
    positives = sorted(((name, x) for name, x in occurrence.items() if x > 0), key=lambda p: (-p[1], p[0]))
    negative = sum(x for x in occurrence.values() if x < 0)
    gross = sum(x for _, x in positives)
    losses = {name: [float(x["final_loss"]) for x in rows["training"]] for name, rows in tables.items()}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_image_error.csv", primary_rows)
    write_csv(output / "stage_density_summary.csv", grouped)
    result = {"checkpoints": checkpoints, "late_ema_final": final, "images_with_extra_fp": len(positives),
              "images_with_fewer_fp": sum(x < 0 for x in occurrence.values()), "gross_increase_fp": gross,
              "gross_decrease_fp": negative, "top_10": positives[:10],
              "top_5_share_of_gross_increase": ratio(sum(x for _, x in positives[:5]), gross),
              "stage_and_density_late_ema": {name: {f"{s}/{g}": average([stats[e, "ema", s, g] for e in range(36, 41)])
                                                  for s in STAGES for g in GROUPS} for name, stats in groups.items()},
              "late_loss": {name: {"E31-35": mean(x[30:35]), "E36-40": mean(x[35:40]),
                                   "relative_change": mean(x[35:40]) / mean(x[30:35]) - 1,
                                   "E31-40": x[30:40]} for name, x in losses.items()}}
    (output / "evidence_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inspect-local", action="store_true")
    args = parser.parse_args()
    audit(args.output, inspect_local=args.inspect_local)
