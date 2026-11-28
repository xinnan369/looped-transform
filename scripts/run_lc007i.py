"""Execute only the approved one-case native-crop freeze."""

import argparse
import shutil
import signal
from pathlib import Path

import torch

from loopcrowd.experiment import file_sha, write_json
from loopcrowd.training import fit
from scripts.prepare_lc007i import ARM, SCREEN, SEED, audit, recipe
from scripts.run_lc007d import background_snapshot, git, read_json
from scripts.run_lc007h_e40 import stop_at_deadline
from scripts.run_lc007h_training import TrainingBudget, require_training_authorization

CASE = f"{SEED}/{ARM}"
EXPOSURE_FIELDS = ("image_order_sha256", "valid_crop_pixels", "sampled_gt", "overflow_gt", "positive_slots", "negative_slots")


def require_launch_record(owner, plan, digest, revision, tree):
    require_training_authorization(owner, plan, digest, revision, tree)
    for field in ("independent_review", "owner_review_waiver"):
        note = owner.get(field)
        if isinstance(note, str) and note.strip():
            return
    raise ValueError("record the independent review or the owner's explicit waiver")


def expected_trajectory(preflight, size, case=CASE):
    rows = [row for row in preflight["exposure"] if row["crop_size"] == size]
    return {"initial_sha256": preflight["planned_starts"][case]["initial_sha256"],
            "trajectory": [(row["crop_order_sha256"], row["learning_rates_sha256"]) for row in rows],
            "crop_exposure": [{key: row[key] for key in EXPOSURE_FIELDS} for row in rows]}


class NativeCropBudget(TrainingBudget):
    label = "LC-007I native crop"


def execute(manifest, output, plan, revision, tree, *, screen=SCREEN, preflight=audit):
    arm, = plan["order"][str(SEED)]
    case = f"{SEED}/{arm}"
    output.mkdir()
    limits = plan["limits_if_approved"]
    budget = NativeCropBudget(limits, output / "gpu_phases.jsonl", torch.device("cuda:0"))
    progress = {"execution": "in_progress", "code_commit": revision, "source_tree": tree, "current": None,
                "config_sha256": file_sha(screen), "completed": [], "external_download_bytes": 0}
    try:
        with budget.phase("cpu_preflight"):
            if shutil.disk_usage(output).free < 3 * 1024**3:
                raise RuntimeError("insufficient disk for native-crop evidence")
            evidence = preflight(manifest, plan)
            write_json(output / "preflight_cpu.json", evidence)
            resource = evidence["projection"]
            if resource["budget_estimate_seconds"] > limits["gpu_seconds"]:
                raise RuntimeError("frozen projection exceeds the approved wall budget")
        with budget.phase("device_preflight"):
            (output / "background_start.txt").write_text(background_snapshot(), encoding="utf-8")
            if (torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name(budget.device)
                    or not torch.cuda.is_bf16_supported()):
                raise RuntimeError("frozen RTX 3090 BF16 environment is unavailable")
            budget.active = True
            free, total = torch.cuda.mem_get_info(budget.device)
            write_json(output / "device_memory_start.json", {"free_bytes": free, "total_bytes": total})
            if free < resource["headroom_bytes_estimate"]:
                raise RuntimeError("insufficient actual device headroom")
        budget.case = progress["current"] = case
        write_json(output / "queue_progress.json", progress | budget.totals())
        result = fit(recipe(plan), manifest, output / "train" / case, budget.device, guard=budget,
                     expected=expected_trajectory(evidence, plan["train_crop_size"], case))
        if (result["completed_epochs"] != limits["epochs_per_run"] or result["global_step"] != limits["total_updates"]
                or budget.updates != limits["total_updates"]):
            raise RuntimeError("execution differs from the frozen successful update denominator")
        progress["completed"].append(case)
        with budget.phase("cleanup"):
            (output / "background_end.txt").write_text(background_snapshot(), encoding="utf-8")
            free, total = torch.cuda.mem_get_info(budget.device)
            write_json(output / "device_memory_end.json", {"free_bytes": free, "total_bytes": total})
        if budget.totals()["charged_seconds"] > limits["gpu_seconds"]:
            raise RuntimeError("execution exceeds the frozen time budget")
        progress.update(execution="completed", current=None)
    except BaseException as error:
        progress.update(execution="stopped", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "queue_progress.json", progress | budget.totals())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--authorization", type=Path)
    mode.add_argument("--cpu-verify-only", action="store_true")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = read_json(SCREEN)
    if args.cpu_verify_only:
        write_json(args.output, audit(args.manifest, plan))
        return
    revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    owner = read_json(args.authorization)
    require_launch_record(owner, plan, file_sha(SCREEN), revision, tree)
    git("merge-base", "--is-ancestor", plan["parent"], revision)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("the frozen checkout must be clean")
    signal.signal(signal.SIGTERM, stop_at_deadline)
    execute(args.manifest, args.output, plan, revision, tree)


if __name__ == "__main__":
    main()
