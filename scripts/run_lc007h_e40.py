"""Run a frozen, single-sweep E40 EMA diagnostic after review and owner authorization."""

import argparse
from collections import Counter
import json
from math import isclose
from pathlib import Path
import random
import signal

import numpy as np
import torch

from loopcrowd.accuracy_diagnosis import density, density_bin
from loopcrowd.calibration import count_match_metrics, json_digest, rank_metrics
from loopcrowd.data import load_rgb
from loopcrowd.experiment import file_sha, make_model, state_sha, write_json
from loopcrowd.grouped_validation import summarize
from loopcrowd.identity_diagnostic import confidence_association, count_identity_details, gt_transitions, matched_identities
from loopcrowd.inference import predict_image_depth
from loopcrowd.manifest import read_manifest
from loopcrowd.model.refinement import gain_from_uncertainty
from scripts.diagnose_lc007h import G, GROUPS, archives
from scripts.prepare_lc007e import config_for
from scripts.prepare_lc007g import SCREEN as SHIFT_SCREEN
from scripts.run_lc007b import state_digest
from scripts.run_lc007d import ScreenBudget, background_snapshot, git, read_json, require_authorization

SCREEN = Path("configs/lc007h_e40_diagnostic.json")
AUDIT = Path("reports/LC-007H/initialization_audit.json")
COUNT_KEYS = ("predicted", "count_error", "abs_count_error", "local_tp", "local_fp", "local_fn")
METRIC_KEYS = ("images", "gt", "predicted", "signed_count_error", "mae", "rmse", "local_nap", "tp", "fp", "fn")


def require_file(path, sha, size=None):
    if (size is not None and path.stat().st_size != size) or file_sha(path) != sha:
        raise ValueError(f"frozen file changed: {path}")


def frozen_inputs(manifest, screen):
    require_file(AUDIT, screen["audit_sha256"])
    require_file(G / "epoch_index.csv", screen["archive_index_sha256"])
    audit = read_json(AUDIT)
    data = audit["data"]
    require_file(manifest, data["manifest_sha256"])
    records = read_manifest(manifest, splits=("val",))
    if ([r.image_id for r in records] != data["ordered_val_ids"]
            or len(records) != screen["limits_if_approved"]["images_per_run"]
            or sum(len(r.points) for r in records) != screen["gt_rows_per_case"]):
        raise ValueError("frozen B validation order or GT coverage differs")
    tables = archives()
    if set(audit["diagnostic_checkpoints"]) != set(screen["order"]):
        raise ValueError("frozen checkpoint coverage differs")
    return audit, records, tables


def reconstruct(meta, config, data_sha, source_commit, screen):
    path = Path(meta["path"])
    require_file(path, meta["sha256"], meta["bytes"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    epoch, step = screen["epoch"], screen["ema_step"]
    ema = payload["ema"]
    if (meta["epoch"] != epoch or meta["ema_step"] != step or len(payload["history"]) != epoch
            or payload["history"][-1]["epoch"] != epoch or payload["history"][-1]["ema_step"] != step
            or type(ema["step"]) is not int or ema["step"] != step or payload["global_step"] != step
            or ema["decay"] != config["ema_decay"] or payload["data_sha256"] != data_sha
            or payload["git_commit"] != source_commit or json_digest(config) != meta["config_sha256"]
            or payload["config"] != config):
        raise ValueError("checkpoint config, E40, EMA or source provenance differs")
    with torch.random.fork_rng(devices=[]):
        model = make_model(config, initialize=False)
    live = payload["model"]
    named = dict(model.named_parameters())
    if ema["shadow"].keys() != named.keys() or any(
        value.dtype != torch.float32 or value.shape != named[name].shape or not torch.isfinite(value).all()
        for name, value in ema["shadow"].items()
    ):
        raise ValueError("invalid unique FP32 EMA parameters")
    model.load_state_dict(live | ema["shadow"], strict=True)
    for name, value in model.named_buffers():
        if not torch.equal(value, live[name]):
            raise ValueError(f"live checkpoint buffer not preserved: {name}")
    return model.eval(), payload["history"][-1]["ema_validation"]


def e40_rows(rows, epoch, keys):
    selected = [row for row in rows if int(row["epoch"]) == epoch and row["policy"] == "ema"]
    result = {tuple(row[k] for k in keys): row for row in selected}
    if len(result) != len(selected):
        raise ValueError("duplicate archived E40 row")
    return result


def image_parity(points, predictions, threshold):
    owned = predictions.count_candidates()
    ordinary, = count_match_metrics(points, owned[owned[:, 2] > threshold], (threshold,))
    identity = count_identity_details(points, predictions, threshold)
    tp, fp, fn = (len(identity[k]) for k in ("matched_ranks", "false_positive_ranks", "unmatched_gt_ids"))
    expected = (tp + fp, fp - fn, abs(fp - fn), tp, fp, fn)
    if tuple(ordinary[key] for key in COUNT_KEYS) != expected:
        raise ValueError("ordinary and identity count metrics differ")
    scores, assigned = rank_metrics(points, predictions)
    ranked = matched_identities(points, predictions)
    if (not np.array_equal(scores, ranked["candidates"][:, 2])
            or not np.array_equal(assigned, ranked["gt_ids"] >= 0)
            or not np.array_equal(ranked["candidate_rows"], np.argsort(-predictions.candidates[:, 2], kind="stable"))):
        raise ValueError("ordinary and identity AP ranking differ")
    return ordinary, (scores, assigned), identity


def group_parity(rows, ranked, archive, saved, tolerance):
    actual = summarize(rows, ranked)
    old_ap, new_ap = saved["nap_local_v1"], actual["nap_local_v1"]
    if old_ap["ap_by_delta"].keys() != new_ap["ap_by_delta"].keys() or any(
            not isclose(old_ap["ap_by_delta"][key], value, abs_tol=tolerance, rel_tol=0)
            for key, value in new_ap["ap_by_delta"].items()):
        raise ValueError("archived E40 threshold AP differs")
    values = (actual["images"], actual["expected_heads"], actual["predicted_heads_on_labeled"],
              actual["signed_count_error"], actual["count"]["mae"], actual["count"]["rmse"], new_ap["nap"],
              actual["local_tp"], actual["local_fp"], actual["local_fn"])
    for key, value in zip(METRIC_KEYS, values, strict=True):
        exact = key not in ("mae", "rmse", "local_nap")
        if (exact and value != int(archive[key])) or (
            not exact and not isclose(value, float(archive[key]), abs_tol=tolerance, rel_tol=0)
        ):
            raise ValueError(f"archived E40 grouped {key} differs: {value} versus {archive[key]}")
    return dict(zip(METRIC_KEYS, values, strict=True)) | {"ap_by_delta": new_ap["ap_by_delta"]}


def ambiguity_bins(association, screen):
    boundaries = screen["uncertainty_boundaries"]
    gt_bin = np.searchsorted(boundaries, association["gt_midpoint_uncertainty"], side="right")
    pred_bin = np.searchsorted(boundaries, association["midpoint_candidate_uncertainty"], side="right")
    missed = ~association["gt_matched_midpoint"]
    recovered = missed & association["gt_matched_final"]
    fp = association["midpoint_candidate_gt_ids"] < 0
    confident_fp = fp & (association["midpoint_score_margin"] + screen["count_threshold"]
                         >= screen["high_confidence_fp_score"])
    kinds = dict(zip(association["midpoint_false_positive_ranks"], association["midpoint_false_positive_kind"], strict=True))
    return [dict(gt=int((gt_bin == i).sum()), fn=int(((gt_bin == i) & missed).sum()),
                 recovered=int(((gt_bin == i) & recovered).sum()), candidates=int((pred_bin == i).sum()),
                 fp=int(((pred_bin == i) & fp).sum()), high_confidence_fp=int(((pred_bin == i) & confident_fp).sum()),
                 fp_categories=dict(Counter(kinds[rank] for rank in np.flatnonzero((pred_bin == i) & fp))))
            for i in range(len(boundaries) + 1)]


def gain_alignment(maps):
    aligned, placebo = [], []
    for score, support in maps:
        score, support = torch.from_numpy(score)[None], torch.from_numpy(support)[None]
        aligned.append(gain_from_uncertainty(score, support, mode="guided")[support].numpy())
        placebo.append(gain_from_uncertainty(score, support, mode="placebo")[support].numpy())
    a, b = np.concatenate(aligned).astype(float), np.concatenate(placebo).astype(float)
    if not np.array_equal(np.sort(a), np.sort(b)) or not len(a):
        raise ValueError("invalid placebo gain multiset")
    return {"cells": len(a), "changed": int(np.count_nonzero(a != b)), "min": float(a.min()),
            "max": float(a.max()), "mean": float(a.mean()), "sd": float(a.std()),
            "pearson": float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None}


def sample(model, image, record, config, device, group, screen, image_archive):
    maps = []
    outputs, _ = predict_image_depth(model, image, config, device, export_identity=True, midpoint_maps=maps)
    if list(outputs) != screen["stages"]:
        raise ValueError("E40 stage coverage differs")
    rows, ranked, identities = {}, {}, {}
    for stage, predictions in outputs.items():
        ordinary, ranked[stage], identities[stage] = image_parity(record.points, predictions, screen["count_threshold"])
        old = image_archive.pop((stage, record.image_id))
        if (old["density_group"] != group or int(old["gt"]) != len(record.points)
                or any(value != int(old[key]) for key, value in ordinary.items())):
            raise ValueError(f"archived E40 image parity failed at {stage}/{record.image_id}")
        rows[stage] = {"image_id": record.image_id, "gt": len(record.points), "density_group": group, "matches": ordinary}
    association = confidence_association(record.points, outputs["early"], outputs["final"], maps, screen["count_threshold"])
    details = {"image_id": record.image_id, "density_group": group, "counts": rows, "unscorable_gt": 0,
               "bins": ambiguity_bins(association, screen), "gain": gain_alignment(maps),
               "stage3_to_final": gt_transitions(identities["stage3"], identities["final"]),
               "midpoint": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in association.items()}}
    return rows, ranked, details


def summarize_bins(details, group, size):
    result = []
    for index in range(size):
        selected = [row["bins"][index] for row in details if group == "all" or row["density_group"] == group]
        keys = ("gt", "fn", "recovered", "candidates", "fp", "high_confidence_fp")
        counts = {k: sum(bucket[k] for bucket in selected) for k in keys}
        kinds = Counter()
        for bucket in selected:
            kinds.update(bucket["fp_categories"])
        result.append(counts | {"fp_categories": dict(kinds),
                                "fn_rate": counts["fn"] / counts["gt"] if counts["gt"] else None,
                                "recovery_rate": counts["recovered"] / counts["fn"] if counts["fn"] else None,
                                "fp_rate": counts["fp"] / counts["candidates"] if counts["candidates"] else None})
    return result


def rng_digest(device):
    kind, values, position, has_gauss, gaussian = np.random.get_state()
    numpy_state = (kind, torch.from_numpy(values.astype(np.int64)), position, has_gauss, gaussian)
    return state_digest([random.getstate(), numpy_state, torch.get_rng_state(),
                         torch.cuda.get_rng_state(device) if device.type == "cuda" else None])


def run_case(label, audit, records, tables, screen, manifest, output, budget):
    seed_text, arm = label.split("/")
    seed = int(seed_text)
    meta = audit["diagnostic_checkpoints"][label]
    data = audit["data"]
    config = config_for(read_json(SHIFT_SCREEN), seed, arm)
    if (config["count_threshold"], config["localization_min_score"]) != (screen["count_threshold"], screen["ap_floor"]):
        raise ValueError("frozen score thresholds differ")
    before_rng = rng_digest(budget.device)
    model, saved = reconstruct(meta, config, data["manifest_sha256"], audit["historical_execution_commit"], screen)
    model.to(budget.device)
    baseline = state_sha(model)
    if rng_digest(budget.device) != before_rng:
        raise ValueError("model construction changed RNG")
    images = e40_rows(tables[seed, arm, "images"], screen["epoch"], ("stage", "image_id"))
    groups = e40_rows(tables[seed, arm, "metrics"], screen["epoch"], ("stage", "density_group"))
    rows, ranked, details = [], [], []
    with (output / "images.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            budget.check("validation_image")
            group = density_bin(density(record), data["train_only_density_tertiles"])
            with load_rgb(record, manifest.parent) as image:
                image_rows, ranks, info = sample(model, image, record, config, budget.device, group, screen, images)
            stream.write(json.dumps(info, allow_nan=False) + "\n")
            stream.flush()
            rows.append(image_rows)
            ranked.append(ranks)
            details.append({k: info[k] for k in ("bins", "image_id", "density_group", "gain")})
    if images or rng_digest(budget.device) != before_rng or state_sha(model) != baseline:
        raise ValueError("image coverage, RNG or EMA/live-BN tensor invariance failed")
    require_file(Path(meta["path"]), meta["sha256"], meta["bytes"])
    metrics = {}
    for stage in screen["stages"]:
        metrics[stage] = {}
        for group in GROUPS:
            idx = [i for i, row in enumerate(rows) if group == "all" or row[stage]["density_group"] == group]
            saved_group = saved[stage] if group == "all" else saved[stage]["density_groups"][group]
            metrics[stage][group] = group_parity([rows[i][stage] for i in idx], [ranked[i][stage] for i in idx],
                                                groups.pop((stage, group)), saved_group,
                                                screen["archive_float_absolute_tolerance"])
    if groups:
        raise ValueError("archived E40 metric groups left unchecked")
    bins = {g: summarize_bins(details, g, len(screen["uncertainty_boundaries"]) + 1) for g in GROUPS}
    if sum(row["gt"] for row in bins["all"]) != screen["gt_rows_per_case"]:
        raise ValueError("uncertainty bins do not account for all GT")
    budget.check()
    write_json(output / "summary.json", {"case": label, "parity": "pass", "checkpoint": meta,
               "model_state_sha256": baseline, "rng_sha256": before_rng, "metrics": metrics, "uncertainty": bins,
               "gains": [row["gain"] | {k: row[k] for k in ("image_id", "density_group")} for row in details],
               "images_sha256": file_sha(output / "images.jsonl"), "unscorable_gt": 0})


def execute(manifest, output, screen, revision):
    output.mkdir()
    budget = ScreenBudget(screen["limits_if_approved"], output / "gpu_phases.jsonl", torch.device("cuda:0"))
    budget.label = "LC-007H E40"
    progress = {"status": "running", "code": revision, "config_sha256": file_sha(SCREEN),
                "external_download_bytes": 0, "completed": []}
    try:
        with budget.phase("cpu_preflight"):
            audit, records, tables = frozen_inputs(manifest, screen)
        with budget.phase("device_preflight"):
            (output / "background_start.txt").write_text(background_snapshot(), encoding="utf-8")
            if torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name(budget.device) or not torch.cuda.is_bf16_supported():
                raise RuntimeError("the historical BF16 RTX 3090 environment is unavailable")
            torch.backends.cudnn.benchmark = False
            budget.active = True
            free, total = torch.cuda.mem_get_info(budget.device)
            progress.update(device_free_bytes=free, device_total_bytes=total)
        for label in screen["order"]:
            budget.case = label
            progress["current"] = label
            write_json(output / "progress.json", progress | budget.totals())
            case_output = output / label
            case_output.mkdir(parents=True)
            with budget.phase("E40_ema"):
                run_case(label, audit, records, tables, screen, manifest, case_output, budget)
            progress["completed"].append(label)
        with budget.phase("cleanup"):
            (output / "background_end.txt").write_text(background_snapshot(), encoding="utf-8")
        if budget.totals()["charged_seconds"] > screen["limits_if_approved"]["gpu_seconds"]:
            raise RuntimeError("charged GPU time exceeded")
        progress.update(status="completed", current=None)
    except BaseException as error:
        progress.update(status="blocked", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "progress.json", progress | budget.totals())


def verify_cpu(manifest, output, screen):
    audit, records, tables = frozen_inputs(manifest, screen)
    results = {}
    rng = rng_digest(torch.device("cpu"))
    for label in screen["order"]:
        seed, arm = label.split("/")
        meta = audit["diagnostic_checkpoints"][label]
        model, _ = reconstruct(meta, config_for(read_json(SHIFT_SCREEN), int(seed), arm),
                               audit["data"]["manifest_sha256"], audit["historical_execution_commit"], screen)
        images = e40_rows(tables[int(seed), arm, "images"], screen["epoch"], ("stage", "image_id"))
        groups = e40_rows(tables[int(seed), arm, "metrics"], screen["epoch"], ("stage", "density_group"))
        if set(images) != {(stage, r.image_id) for stage in screen["stages"] for r in records} or set(groups) != {
            (stage, group) for stage in screen["stages"] for group in GROUPS
        }:
            raise ValueError(f"archived E40 coverage differs for {label}")
        results[label] = {"model_state_sha256": state_sha(model), "checkpoint_sha256": meta["sha256"],
                          "image_rows": len(images), "group_rows": len(groups)}
    if rng_digest(torch.device("cpu")) != rng:
        raise ValueError("checkpoint verification changed RNG")
    write_json(output, {"status": "cpu_verified", "config_sha256": file_sha(SCREEN), "cases": results,
                        "checkpoint_inference": "not_run"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--authorization", type=Path)
    mode.add_argument("--cpu-verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = read_json(SCREEN)
    if args.cpu_verify_only:
        verify_cpu(args.manifest, args.output, screen)
        return
    revision = git("rev-parse", "HEAD")
    authorization = read_json(args.authorization)
    require_authorization(authorization, screen, file_sha(SCREEN), revision)
    if authorization["approval_scope"] != screen["scope"]:
        raise ValueError("authorization does not cover the E40 read-only scope")
    git("merge-base", "--is-ancestor", screen["parent"], revision)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("a clean frozen worktree is required")
    signal.signal(signal.SIGTERM, stop_at_deadline)
    execute(args.manifest, args.output, screen, revision)


def stop_at_deadline(signum, frame):
    raise TimeoutError("external LC-007H wall-clock deadline reached")


if __name__ == "__main__":
    main()
