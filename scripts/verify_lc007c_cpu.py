"""Measure generated two-call fixtures and archive equivalence and budget estimates."""

import argparse
import platform
from statistics import median
from time import perf_counter
from pathlib import Path

import numpy as np
import torch

from loopcrowd.ema import preserve_training_context
from loopcrowd.experiment import write_json
from loopcrowd.gradient_diagnosis import TOLERANCES, WEIGHT, diagnose_two_calls
from loopcrowd.inference import autocast
from loopcrowd.model import LoopCrowd
from loopcrowd.targets import point_targets
from loopcrowd.training import supervised_point_loss
from scripts.archive_lc007c import ARCHIVE, OUTPUT, SEEDS, SHARED, UNSHARED
from scripts.run_lc007a import read_csv


def one_backward(model, images, target):
    # Timing comparator only. This generated fixture performs no optimizer update.
    with preserve_training_context(model, images.device):
        model.train()
        with autocast(images.device, "bf16"):
            loss = supervised_point_loss(model, images, target, {"early_loss_weight": WEIGHT, "density_weight": 0})
        torch.autograd.grad(loss, tuple(model.parameters()))


def synthetic_case(variant, side):
    torch.manual_seed(70703)
    model = LoopCrowd(variant).train()
    images = torch.randn(2, 3, side, side)
    sample = point_targets([[4., 7.], [side / 2, side / 2 + 1], [side - 3., side - 7.]], side, side)
    target = {key: torch.stack((value, value)) for key, value in sample.items()}
    buffers = dict(model.named_buffers())
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    timings = {"standard": [], "probe": []}
    diagnostic = None
    for index in range(4):
        # Reset the synthetic timing comparator's live BN buffers after each step.
        start = perf_counter()
        one_backward(model, images, target)
        reference_seconds = perf_counter() - start
        with torch.no_grad():
            for key, value in buffers.items():
                value.copy_(saved[key])
        start = perf_counter()
        diagnostic = diagnose_two_calls(model, images, target, precision="bf16")
        probe_seconds = perf_counter() - start
        if index:
            timings["standard"].append(reference_seconds)
            timings["probe"].append(probe_seconds)
    torch.testing.assert_close(model.state_dict(), saved, atol=0, rtol=0)
    return {"variant": variant, "synthetic_batch": list(images.shape), "precision": "bf16", "seconds": timings,
            "probe_to_single_backward_ratio": median(timings["probe"]) / median(timings["standard"]),
            "outputs_max_abs": diagnostic["outputs_max_abs"], "losses_max_abs": diagnostic["losses_max_abs"],
            "native_vs_expanded_gradient_checks": diagnostic["gradient_equivalence"],
            "live_model_and_bn_unchanged": True}


def estimate(cases):
    historical_ms = [float(row["mean_train_batch_ms"]) for seed in SEEDS for arm in (SHARED, UNSHARED)
                     for row in read_csv(ARCHIVE / "epochs" / str(seed) / arm / "training.csv")]
    ms = np.quantile(historical_ms, (.10, .50, .90)).tolist()
    ratios = [row["probe_to_single_backward_ratio"] for row in cases]
    kernel_proxy = [96 * ms[0] * min(ratios) / 1000, 96 * ms[2] * max(ratios) / 1000]
    return {
        "historical_epoch_means": len(historical_ms), "historical_epoch_mean_batch_ms_p10_p50_p90": ms,
        "cpu_synthetic_probe_to_single_backward_ratio_min_max": [min(ratios), max(ratios)],
        "96_case_training_cost_proxy_seconds": kernel_proxy,
        "planning_upper_seconds": 2 * kernel_proxy[1] + 60,
        "formula": "2 × upper historical-step-scaled synthetic ratio + 60 seconds for I/O, preflight and finalization",
        "limits": "different device, input size, precision dispatch and allocator; no GPU timing or memory was measured",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cases = [synthetic_case(variant, side) for side in (64, 128) for variant in ("shared_2", "unshared_2")]
    result = {
        "real_checkpoint_gradient_execution": "not_run_pending_owner_gpu_approval",
        "synthetic_cpu_execution": "passed", "generated_data_only": True, "torch": str(torch.__version__),
        "python": platform.python_version(), "tolerances_abs_rel": TOLERANCES, "cases": cases,
        "resource_projection_unvalidated": estimate(cases),
    }
    write_json(args.output / "gradient_equivalence.json", result)
    print(result["resource_projection_unvalidated"])


if __name__ == "__main__":
    main()
