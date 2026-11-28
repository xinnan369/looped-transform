"""Reconcile the native-crop comparison before a separately approved run."""

from pathlib import Path

import torch

from loopcrowd.experiment import file_sha, make_model
from scripts.archive_lc007c import check_training
from scripts.prepare_lc007i import audit as native_audit, recipe, reference_tables
from scripts.profile import forward_costs
from scripts.run_lc007a import read_csv
from scripts.run_lc007d import read_json
from scripts.run_lc007i import EXPOSURE_FIELDS

SCREEN = Path("configs/lc007k_unshared_native768.json")
SHARED = Path("reports/LC-007I/training_results")
SHARED_ARM, SHARED_512, UNSHARED_ARM = "S4-shift-768", "S4-shift", "U4-shift-768"


def shared_control(plan):
    return crop_control(SHARED, plan["s768_sha256"], SHARED_ARM)


def crop_control(folder, fingerprints, arm):
    for name, digest in fingerprints.items():
        if file_sha(folder / name) != digest:
            raise ValueError(f"frozen 768 source differs: {name}")
    tables = {}
    for row in read_csv(folder / "epoch_index.csv"):
        path = folder / row["path"]
        if path.stat().st_size != int(row["bytes"]) or file_sha(path) != row["sha256"]:
            raise ValueError(f"frozen shared 768 table differs: {path}")
        tables[row["kind"]] = read_csv(path)
        if len(tables[row["kind"]]) != int(row["rows"]):
            raise ValueError(f"frozen shared 768 row count differs: {path}")
    check_training(tables["training"])
    evidence = read_json(folder / "run_evidence.json")
    status = read_json(folder / "research_decision.json")
    case = evidence["case"]
    if (evidence["execution"]["execution"] != "completed" or case["status"] != "completed"
            or case["seed"] != 42 or case["arm"] != arm or case["successful_updates"] != 7200
            or status["decision"] != "fail"):
        raise ValueError("shared 768 provenance or accepted negative outcome differs")
    return tables, evidence


def assert_paired_exposure(exposure, training):
    for expected, actual in zip(exposure, training, strict=True):
        keys = ("epoch", "crop_order_sha256", "learning_rates_sha256", *EXPOSURE_FIELDS)
        for key in keys:
            if str(expected[key]) != str(actual[key]):
                raise ValueError(f"native 768 differs from actual shared exposure at E{expected['epoch']} {key}")


def audit(manifest, plan):
    observed, historical = shared_control(plan)
    s512 = reference_tables(plan | {"historical_reference": SHARED_512})
    evidence = native_audit(manifest, plan, screen=SCREEN)
    large = [r for r in evidence["exposure"] if r["crop_size"] == plan["train_crop_size"]]
    assert_paired_exposure(large, observed["training"])
    shared = read_json(SHARED / "preflight_cpu.json")
    unshared = evidence["planned_starts"][f"42/{UNSHARED_ARM}"]
    common = shared["planned_starts"][f"42/{SHARED_ARM}"]
    if (any(common[k] != unshared[k] for k in ("common_sha256", "encoder_sha256", "decoder_sha256"))
            or common["block_sha256"][0] != unshared["block_sha256"][0]
            or len(set(unshared["block_sha256"])) != 4 or unshared["parameters"] != 2931206
            or common["parameters"] != 1596614):
        raise ValueError("common initialization or historical independent blocks differ")
    for key in ("manifest_sha256", "pretrained_weight_sha256", "ordered_val_ids", "train_only_density_tertiles"):
        if shared["data"][key] != evidence["data"][key]:
            raise ValueError(f"shared 768 split differs at {key}")
    # S512/U512 historically used the same sampler and crop rule at this seed.
    legacy = [r for r in evidence["exposure"] if r["crop_size"] == 512]
    if any(a["crop_order_sha256"] != b["crop_order_sha256"] or a["learning_rates_sha256"] != b["learning_rates_sha256"]
           for a, b in zip(legacy, s512["training"], strict=True)):
        raise ValueError("same-seed archived 512 trajectories differ")
    if (evidence["dense_two_read_flops"] != shared["dense_two_read_flops"]
            or evidence["dense_four_read_512_flops"] != shared["dense_four_read_512_flops"]):
        raise ValueError("nominal dense matched-size two/four-read computation differs")
    final_costs = {}
    for variant in ("shifted_shared_4", "shifted_unshared_4"):
        model = make_model(recipe(plan) | {"variant": variant}, initialize=False).eval()
        final_costs[variant] = forward_costs(model, torch.zeros(1, 3, 512, 512))
    if final_costs["shifted_shared_4"] != final_costs["shifted_unshared_4"]:
        raise ValueError("nominal matched-size final-only inference cost differs")
    phases = evidence["projection"]["historical_phase_seconds"]
    measured = {name: sum(p["seconds"] for p in historical["phases"] if p["phase"] == name) for name in phases}
    factor = max(evidence["projection"]["area_ratio"], evidence["projection"]["dense_ratio"])
    planned = {name: max(measured[name], secs * (factor if name == "training" else 1)) for name, secs in phases.items()}
    base = sum(planned.values())
    evidence["projection"].update(shared768_phase_seconds=measured, shared768_charged_seconds=historical["execution"]["charged_seconds"],
        projected_phase_seconds=planned, base_seconds=base, allowance_seconds=.25 * base + 120,
        budget_estimate_seconds=1.25 * base + 120)
    evidence["shared768_checkpoint"] = historical["case"]["retained_checkpoint"]
    evidence["shared768_execution_commit"] = historical["execution"]["code_commit"]
    evidence["dense_final_only_512_flops"] = final_costs
    evidence["actual_shared768_exposure_matched_epochs"] = len(large)
    return evidence
