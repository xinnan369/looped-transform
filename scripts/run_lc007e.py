"""Execute the reviewed K=4 queue only with a separately traceable owner approval."""

import argparse
import shutil
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from loopcrowd.experiment import file_sha, make_model, seed_all, write_json
from loopcrowd.inference import autocast
from loopcrowd.training import fit
from scripts.prepare_lc007e import SCREEN, audit, config_for, history, trajectory_columns
from scripts.run_lc007d import ScreenBudget, background_snapshot, git, read_json, require_authorization


class DepthBudget(ScreenBudget):
    label = "LC-007E"
    cleanup_reserve_seconds = 15


@torch.inference_mode()
def latency(screen, budget, *, load_config=config_for, modes=("final_only", "staged")):
    image = torch.zeros(1, 3, 512, 512, device=budget.device)
    arms = screen["order"]["42"]
    samples = {arm: {mode: [] for mode in modes} for arm in arms}
    models = {}
    for arm in arms:
        seed_all(42)
        models[arm] = make_model(load_config(screen, 42, arm), initialize=True).to(budget.device).eval()
    with autocast(budget.device, "bf16"):
        for repetition in range(6):
            order = arms[repetition % len(arms):] + arms[:repetition % len(arms)]
            for arm in order:
                model = models[arm]
                modes = tuple(samples[arm])
                for mode in (modes if repetition % 2 == 0 else modes[::-1]):
                    budget.check()
                    torch.cuda.synchronize(budget.device)
                    start = perf_counter()
                    output = ({"final": model(image)} if mode == "final_only"
                              else model.forward_depth_stages(image, diagnostic=mode != "midpoint_final"))
                    torch.cuda.synchronize(budget.device)
                    if repetition:
                        samples[arm][mode].append(1000 * (perf_counter() - start))
                    if not all(torch.isfinite(t).all() for fields in output.values() for t in fields.values()):
                        raise ValueError("nonfinite synthetic depth probe output")
    return {arm: {mode: {"samples_ms": values, "median_ms": median(values)}
                  for mode, values in modes.items()} for arm, modes in samples.items()}


def execute(manifest, output, screen, revision, *, audit_inputs=audit, history_tables=history, budget_type=DepthBudget,
            projection_keys=("upper_with_15pct_contingency_seconds", "peak_reserved_scenario_2x_bytes"),
            latency_modes=("final_only", "staged"), free_disk_gib=3):
    if shutil.disk_usage(output.parent).free < free_disk_gib * 1024**3:
        raise RuntimeError(f"{budget_type.label} requires {free_disk_gib} GiB free for checkpoints and evidence")
    output.mkdir()
    progress = {"execution": "in_progress", "code_commit": revision, "completed": []}
    budget = None
    limits = screen["limits_if_approved"]
    try:
        evidence = audit_inputs(manifest, screen)
        write_json(output / "preflight_cpu.json", evidence)
        projection = evidence["resource_projection"]
        if (projection[projection_keys[0]] > limits["gpu_seconds"]
                or projection[projection_keys[1]] > limits["peak_reserved_bytes"]):
            raise RuntimeError("CPU resource projection exceeds the proposed limits")
        tables = history_tables()
        device = torch.device("cuda:0")
        budget = budget_type(limits, output / "gpu_phases.jsonl", device)
        try:
            with budget.phase("preflight"):
                (output / "background_start.txt").write_text(background_snapshot(), encoding="utf-8")
                if torch.cuda.device_count() != 1:
                    raise RuntimeError("one visible RTX 3090 is required")
                budget.active = True
                if "3090" not in torch.cuda.get_device_name(device) or not torch.cuda.is_bf16_supported():
                    raise RuntimeError("one visible BF16-capable RTX 3090 is required")
                free, total = torch.cuda.mem_get_info(device)
                if free < limits["peak_reserved_bytes"]:
                    raise RuntimeError("existing jobs leave less than 2 GiB free")
                torch.cuda.set_per_process_memory_fraction(limits["peak_reserved_bytes"] / total, device)
                write_json(output / "synthetic_gpu_latency.json", latency(screen, budget, modes=latency_modes))
                torch.cuda.empty_cache()
            for seed_text, arms in screen["order"].items():
                for arm in arms:
                    if git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "--untracked-files=no"):
                        raise RuntimeError("reviewed worktree changed during execution")
                    budget.case = f"{seed_text}/{arm}"
                    progress["current"] = budget.case
                    write_json(output / "queue_progress.json", progress | budget.totals())
                    seed = int(seed_text)
                    config = config_for(screen, seed, arm)
                    reference = screen["arms"][arm]["comparison"]
                    expected = {"initial_sha256": evidence["planned_starts"][budget.case]["initial_sha256"],
                                "trajectory": trajectory_columns(tables[seed, reference, "training"])}
                    result = fit(config, manifest, output / "train" / seed_text / arm, device, guard=budget, expected=expected)
                    if result["completed_epochs"] != 40 or result["global_step"] != 7200:
                        raise RuntimeError("incorrect frozen horizon")
                    progress["completed"].append(budget.case)
                    write_json(output / "queue_progress.json", progress | budget.totals())
        finally:
            if budget.active:
                with budget.phase("cleanup"):
                    torch.cuda.empty_cache()
                    (output / "background_end.txt").write_text(background_snapshot(), encoding="utf-8")
        totals = budget.totals()
        if (len(progress["completed"]) != limits["runs"] or budget.updates != limits["total_updates"]
                or totals["charged_seconds"] > limits["gpu_seconds"] or budget.reserved > limits["peak_reserved_bytes"]):
            raise RuntimeError("final resource ledger or update denominator differs")
        progress.update(execution="completed", current=None)
    except BaseException as error:
        progress.update(execution="stopped", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "queue_progress.json", progress | (budget.totals() if budget is not None else {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = read_json(SCREEN)
    revision = git("rev-parse", "HEAD")
    require_authorization(read_json(args.authorization), screen, file_sha(SCREEN), revision)
    git("merge-base", "--is-ancestor", screen["parent"], revision)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("a clean frozen worktree is required")
    execute(args.manifest, args.output, screen, revision)


if __name__ == "__main__":
    main()
