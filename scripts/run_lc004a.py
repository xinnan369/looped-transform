"""Run the preregistered LC-004A local experiment on one CUDA device."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loopcrowd.diagnostics import backbone_bn_buffers, compare_prefixes, describe_bn, evaluate_prefix
from loopcrowd.experiment import DEFAULT_CONFIG, file_sha, load_checkpoint, make_model, read_config, write_json
from loopcrowd.manifest import read_manifest
from loopcrowd.model import pretrained_path
from scripts.run_lc003 import MANIFEST_SHA, WEIGHT_SHA, load_json, run_logged, stability

BASELINES = ("cnn_only", "shared_1", "shared_2", "unshared_2")
GAINS = ("shared_2_tied_gain", "shared_2_stage_gain")
SEEDS = (42, 123, 2026)
FROZEN_EVIDENCE = Path("reports/LC-003/training_history.json")
TRAINING_CODE = "34e310976d1bdbd0eab7855eb02bf087cba86638"


def choose_bn(comparisons):
    for variant in ("cnn_only", "shared_2"):
        train, frozen = (comparisons[arm][variant] for arm in ("train", "frozen"))
        if not (frozen["pass"] and frozen["late_mae"] < train["late_mae"]
                and frozen["late_nap"] >= train["late_nap"] - 0.005):
            return "train"
    return "frozen"


def gain_gate(results, last_prefix_delta):
    stage = results["shared_2_stage_gain"]
    comparators = {name: results[name] for name in ("shared_2", "shared_2_tied_gain", "shared_1")}
    reductions = {
        name: 1 - stage["late_mae"] / arm["late_mae"] if arm["late_mae"] else 0.0
        for name, arm in comparators.items()
    }
    nap_delta = stage["late_nap"] - max(arm["late_nap"] for arm in comparators.values())
    contribution = (
        last_prefix_delta["mae_delta"] < 0
        and last_prefix_delta["improved"] > last_prefix_delta["degraded"]
    )
    checks = {
        "stable": stage["pass"],
        "mae_reductions_ge_3_percent": all(value >= 0.03 for value in reductions.values()),
        "nap_delta_ge_minus_0_005": nap_delta >= -0.005,
        "epoch40_same_checkpoint_prefix_contribution": contribution,
    }
    return {
        "checks": checks, "mae_reduction": reductions, "nap_difference_from_best_comparator": nap_delta,
        "epoch40_two_minus_one": last_prefix_delta,
        "pass": all(checks.values()), "additional_seeds": "in_progress" if all(checks.values()) else "not_run",
    }


def verify_workload(a, b):
    fields = ("crops", "sampled_gt", "overflow_gt", "positive_slots", "negative_slots", "learning_rates")
    if any(left[k] != right[k] for left, right in zip(a, b, strict=True) for k in fields):
        raise ValueError("matched arms have different crop coverage or applied LR trajectories")


def assert_archived(source, archived, config, manifest_sha):
    run_file = source / "run_manifest.json"
    run = load_json(run_file)
    history_file = source / "history.json"
    if file_sha(history_file) != archived["local_history"]["sha256"]:
        raise ValueError(f"frozen run history differs: {source}")
    if file_sha(source.with_suffix(".log")) != archived["training_log"]["sha256"]:
        raise ValueError(f"frozen run log differs: {source}")
    if (
        run != archived["provenance"] or run["git_commit"] != TRAINING_CODE
        or run["tracked_changes"] or run["execution"] != "completed"
        or DEFAULT_CONFIG | run["config"] | {"backbone_bn": "train"} != config
        or run["data"]["sha256"] != manifest_sha or run["weight_initialization"]["sha256"] != WEIGHT_SHA
    ):
        raise ValueError(f"cannot reuse LC-003 arm with different provenance: {source}")
    history = load_json(history_file)
    stability(history)
    for checkpoint in ("best.pt", "last.pt"):
        if file_sha(source / checkpoint) != run["checkpoints"][checkpoint]["sha256"]:
            raise ValueError(f"frozen checkpoint digest differs: {source / checkpoint}")
    return run, history


def assert_metrics(diagnostic, epoch):
    overall = diagnostic["overall"]
    metrics = epoch["validation"]
    if (
        overall["mae"] != metrics["count"]["mae"] or overall["rmse"] != metrics["count"]["rmse"]
        or overall["local_nap"] != metrics["nap_local_v1"] or overall["gt"] != metrics["expected_heads"]
        or overall["predicted"] != metrics["predicted_heads_on_labeled"]
    ):
        raise AssertionError("full-depth validation metrics differ from training archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lc003", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if file_sha(args.manifest) != MANIFEST_SHA or file_sha(pretrained_path()) != WEIGHT_SHA:
        parser.error("local manifest and pretrained file do not match frozen SHA-256 digests")
    records = read_manifest(args.manifest)
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    if len(train) != 270 or len(val) != 30 or any(
        record.points is None or not (args.manifest.parent / record.image_path).is_file()
        for record in train + val
    ):
        parser.error("local training or validation files are missing or do not match the frozen split")
    train_density = [len(r.points) * 1e6 / (r.image_width * r.image_height) for r in train]
    boundaries = np.quantile(train_density, (1 / 3, 2 / 3)).tolist()
    frozen_boundaries = load_json(Path("reports/LC-003/diagnosis.json"))["heads_per_megapixel_train_tertiles"]
    np.testing.assert_allclose(boundaries, frozen_boundaries, rtol=1e-13)
    device = torch.device("cuda")
    configs = {arm: Path(f"configs/lc004a_{arm}_bn.json") for arm in ("train", "frozen")}
    expected = {arm: read_config(path) for arm, path in configs.items()}
    if expected["train"] | {"backbone_bn": "frozen"} != expected["frozen"]:
        raise ValueError("the BN configurations must differ only by the backbone BN policy")

    evidence = load_json(FROZEN_EVIDENCE)
    historical = {}
    for seed in SEEDS:
        for variant in BASELINES:
            source = args.lc003 / "train" / "warmup_cosine" / str(seed) / variant
            archive = evidence["runs"][f"warmup_cosine/{seed}/{variant}"]
            run, history = assert_archived(source, archive, expected["train"] | {"seed": seed, "variant": variant},
                                           MANIFEST_SHA)
            historical[seed, variant] = (source, run, history)
    args.output.mkdir(parents=True)
    reference = backbone_bn_buffers(make_model(expected["train"], initialize=True))
    write_json(args.output / "bn_pretrained.json", describe_bn(reference))
    write_json(args.output / "inputs.json", {
        "manifest": {"path": str(args.manifest), "sha256": MANIFEST_SHA},
        "weights": {"path": str(pretrained_path()), "sha256": WEIGHT_SHA},
        "lc003_archive": {"path": str(FROZEN_EVIDENCE), "sha256": file_sha(FROZEN_EVIDENCE)},
        "train_density_boundaries": frozen_boundaries, "train_images": len(train), "validation_images": len(val),
    })

    def diagnose(source, run, history, destination, prefixes=True):
        results = {}
        for tag, index in (("best", run["best_epoch"] - 1), ("last", -1)):
            file = source / f"{tag}.pt"
            state = load_checkpoint(file, torch.device("cpu"))
            if state["data_sha256"] != MANIFEST_SHA or state["config"] != run["config"]:
                raise ValueError(f"checkpoint metadata mismatch: {file}")
            model = make_model(state["config"], initialize=False)
            model.load_state_dict(state["model"])
            bn = describe_bn(backbone_bn_buffers(model), reference)
            if run["config"].get("backbone_bn") == "frozen":
                assert all(row[key]["max_abs_delta_from_pretrained"] == 0
                           for row in bn.values() for key in ("running_mean", "running_var"))
                assert all(bn[layer]["num_batches_tracked"] == values["num_batches_tracked"]
                           for layer, values in reference.items())
            result = {"checkpoint": run["checkpoints"][file.name], "epoch": history[index]["epoch"],
                      "training_code_sha": run["git_commit"], "config_sha256": run["config_sha256"],
                      "bn": bn}
            if prefixes:
                print(f"diagnose {file}", flush=True)
                model.to(device)
                recurrent = run["config"]["variant"] in ("shared_2", *GAINS)
                calls = range(3) if recurrent else (model.loops,)
                result["prefixes"] = {
                    str(call): evaluate_prefix(model, val, args.manifest.parent, state["config"], device,
                                               frozen_boundaries, call, verify_default=recurrent and call == 2)
                    for call in calls
                }
                assert_metrics(result["prefixes"][str(model.loops)], history[index])
                if recurrent:
                    result["two_minus_one"] = compare_prefixes(result["prefixes"]["1"], result["prefixes"]["2"])
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / f"{tag}.json", result)
            results[tag] = result
        return results

    for (seed, variant), (source, run, history) in historical.items():
        diagnose(source, run, history, args.output / "diagnostics" / "lc003" / str(seed) / variant)

    shared_args = ("--manifest", args.manifest, "--device", "cuda")

    def train_arm(arm, seed, variant):
        output = args.output / "train" / arm / str(seed) / variant
        run_logged(output, "train", (*shared_args, "--config", configs[arm], "--seed", seed, "--variant", variant))
        run, history = load_json(output / "run_manifest.json"), load_json(output / "history.json")
        verify_workload(historical[seed, "cnn_only"][2], history)
        return output, run, history

    bn_ablation = {arm: {} for arm in expected}
    selected_controls = {}
    for variant in BASELINES:
        bn_ablation["train"][variant] = stability(historical[42, variant][2])
        output, run, history = train_arm("frozen", 42, variant)
        bn_ablation["frozen"][variant] = stability(history)
        diagnose(output, run, history, args.output / "bn_buffers" / "frozen" / "42" / variant, prefixes=False)
        selected_controls[variant] = (output, run, history)
    bn_selected = choose_bn(bn_ablation)
    write_json(args.output / "bn_decision.json", {"selected": bn_selected, "ablation": bn_ablation})

    if bn_selected == "train":
        selected_controls = {variant: historical[42, variant] for variant in BASELINES}
    else:
        diagnose(*selected_controls["shared_2"], args.output / "diagnostics" / "frozen" / "42" / "shared_2")
    results = {variant: stability(history) for variant, (_, _, history) in selected_controls.items()}
    for variant in GAINS:
        output, run, history = train_arm(bn_selected, 42, variant)
        results[variant] = stability(history)
        diagnostic = diagnose(output, run, history, args.output / "diagnostics" / bn_selected / "42" / variant)
    decision = gain_gate(results, diagnostic["last"]["two_minus_one"]["overall"])
    write_json(args.output / "gain_decision.json", decision | {"seed42": results})
    for variant in (*BASELINES, *GAINS):
        run_logged(args.output / "profile" / variant, "profile", (
            *shared_args, "--config", "configs/lc002_profile.json", "--backbone-bn", bn_selected,
            "--variant", variant, "--repeats", 9,
        ))
    if decision["pass"]:
        for seed in (123, 2026):
            for variant in (*BASELINES, *GAINS):
                if bn_selected == "train" and variant in BASELINES:
                    continue
                output, run, history = train_arm(bn_selected, seed, variant)
                if variant in GAINS:
                    diagnose(output, run, history, args.output / "diagnostics" / bn_selected / str(seed) / variant)
        decision["additional_seeds"] = "completed"
        write_json(args.output / "gain_decision.json", decision | {"seed42": results})
    print("LC-004A queue completed", flush=True)


if __name__ == "__main__":
    main()
