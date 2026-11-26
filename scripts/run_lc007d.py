"""Run the frozen B screen only after separately recording the owner's GPU authorization."""

import argparse
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from loopcrowd.experiment import file_sha, make_model, read_config, seed_all, write_json
from loopcrowd.inference import autocast
from loopcrowd.training import fit
from scripts.archive_lc007c import read_frozen_files
from scripts.prepare_lc007d import CONFIG, audit
from scripts.run_lc007b import DEFAULT_RESERVES


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_authorization(record, screen, config_sha, revision):
    for field in ("owner_message_ref", "approved_at", "approval_scope"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"separate owner authorization is missing {field}")
    if (record.get("issue") != screen["issue"] or record.get("code_commit") != revision
            or record.get("config_sha256") != config_sha or record.get("limits") != screen["limits_if_approved"]):
        raise ValueError("recorded owner authorization does not match this frozen screen")


class ScreenBudget:
    """Charge wall time from preflight through final synchronization, including failures."""

    label = "LC-007D"
    cleanup_reserve_seconds = 5

    def __init__(self, limits, ledger, device):
        self.limits = limits
        self.ledger = ledger
        self.device = device
        self.start = perf_counter()
        self.active = False
        self.case = "preflight"
        self.updates = self.allocated = self.reserved = 0
        self.longest = {}

    def totals(self):
        if self.active:
            self.allocated = max(self.allocated, torch.cuda.max_memory_allocated(self.device))
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved(self.device))
        return {"charged_seconds": perf_counter() - self.start, "peak_allocated_bytes": self.allocated,
                "peak_reserved_bytes": self.reserved, "successful_updates": self.updates}

    def check(self, operation=None):
        totals = self.totals()
        estimate = max(DEFAULT_RESERVES.get(operation, 0), self.longest.get(operation, 0))
        # Reserve final synchronization and cleanup even when checking a single batch.
        if totals["charged_seconds"] + estimate + self.cleanup_reserve_seconds >= self.limits["gpu_seconds"]:
            raise RuntimeError(f"{self.label} time budget would be exceeded during {operation or self.case}")
        peak_limit = self.limits.get("peak_reserved_bytes")
        if peak_limit is not None and self.reserved > peak_limit:
            raise RuntimeError(f"{self.label} process reserved exceeds the 2 GiB cap")
        if operation == "train_batch" and self.updates >= self.limits["total_updates"]:
            raise RuntimeError(f"{self.label} optimizer-update cap reached")

    def record_update(self):
        self.updates += 1
        if self.updates > self.limits["total_updates"]:
            raise RuntimeError(f"{self.label} optimizer-update cap exceeded")

    @contextmanager
    def phase(self, name):
        if name != "cleanup":
            self.check(name)
        start, failure = perf_counter(), None
        try:
            if self.active:
                torch.cuda.synchronize(self.device)
            yield
            if self.active:
                torch.cuda.synchronize(self.device)
            if name != "cleanup":
                self.check()
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            seconds = perf_counter() - start
            self.longest[name] = max(self.longest.get(name, 0), seconds)
            with self.ledger.open("a", encoding="utf-8") as stream:
                row = {"case": self.case, "phase": name, "seconds": seconds, "failure": failure} | self.totals()
                stream.write(json.dumps(row, allow_nan=False) + "\n")


def background_snapshot():
    executable = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    return subprocess.check_output([executable], text=True, encoding="utf-8", timeout=10)


@torch.inference_mode()
def staged_latency(config, guard):
    device = guard.device
    seed_all(config["seed"])
    model = make_model(config, initialize=True).to(device).eval()
    image = torch.zeros(1, 3, 512, 512, device=device)
    samples = []
    with autocast(device, config["precision"]):
        for index in range(6):
            guard.check()
            torch.cuda.synchronize(device)
            start = perf_counter()
            prediction = model.forward_stages(image)
            torch.cuda.synchronize(device)
            if index:
                samples.append((perf_counter() - start) * 1000)
            if not all(torch.isfinite(value).all() for fields in prediction.values() for value in fields.values()):
                raise ValueError("nonfinite synthetic staged preflight output")
    return {"samples_ms": samples, "median_ms": median(samples)}


def execute(manifest, output, screen, revision):
    # The full metadata/initialization audit runs on CPU before enabling GPU work.
    verified = audit(manifest, screen, profile=False)
    archive = Path(screen["historical_archive"])
    tables = read_frozen_files(archive)
    baseline = read_json(archive / "preflight.json")["frozen"]
    output.mkdir(parents=True)
    write_json(output / "preflight_cpu.json", verified)
    limits = screen["limits_if_approved"]
    device = torch.device("cuda:0")
    budget = ScreenBudget(limits, output / "gpu_phases.jsonl", device)
    progress = {"execution": "in_progress", "completed": [], "code_commit": revision}
    progress_path = output / "queue_progress.json"
    try:
        try:
            with budget.phase("preflight"):
                (output / "background_start.txt").write_text(background_snapshot(), encoding="utf-8")
                if torch.cuda.device_count() != 1:
                    raise ValueError("the frozen screen requires one visible BF16-capable RTX 3090")
                budget.active = True
                if "3090" not in torch.cuda.get_device_name(device) or not torch.cuda.is_bf16_supported():
                    raise ValueError("the frozen screen requires one visible BF16-capable RTX 3090")
                available, total = torch.cuda.mem_get_info(device)
                if available < limits["peak_reserved_bytes"]:
                    raise RuntimeError("existing GPU work leaves less than 2 GiB free")
                torch.cuda.set_per_process_memory_fraction(limits["peak_reserved_bytes"] / total, device)
                latency = {}
                for reference in ("shared_2_aux_025", "unshared_2_aux_025"):
                    config = read_config(Path(baseline["configs"][reference]))
                    latency[reference] = staged_latency(config, budget)
                    write_json(output / "synthetic_gpu_latency.json", latency)
                for arm in screen["order"]["42"]:
                    spec = screen["variants"][arm]
                    config = read_config(Path(baseline["configs"][spec["reference"]]), spec["variant"])
                    latency[arm] = staged_latency(config, budget)
                    write_json(output / "synthetic_gpu_latency.json", latency)
            for seed, order in screen["order"].items():
                for arm in order:
                    budget.case = f"{seed}/{arm}"
                    if git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "--untracked-files=no"):
                        raise ValueError("the frozen worktree changed during the screen")
                    progress["current"] = budget.case
                    write_json(progress_path, progress | budget.totals())
                    spec = screen["variants"][arm]
                    reference = spec["reference"]
                    config = read_config(Path(baseline["configs"][reference]), spec["variant"]) | {
                        "seed": int(seed), "ema_decay": screen["ema_decay"],
                    }
                    expected = {
                        "initial_sha256": verified["planned_gated_initialization"][budget.case]["initial_sha256"],
                        "trajectory": [(row["crop_order_sha256"], row["learning_rates_sha256"])
                                       for row in tables[int(seed), reference, "training"]],
                    }
                    run = fit(config, manifest, output / "train" / seed / arm, device, guard=budget, expected=expected)
                    if (run["completed_epochs"] != limits["epochs_per_run"]
                            or run["global_step"] != limits["epochs_per_run"] * limits["updates_per_epoch"]):
                        raise ValueError("run did not complete its frozen horizon")
                    progress["completed"].append(budget.case)
                    write_json(progress_path, progress | budget.totals())
        finally:
            if budget.active:
                with budget.phase("cleanup"):
                    torch.cuda.empty_cache()
                    (output / "background_end.txt").write_text(background_snapshot(), encoding="utf-8")
        totals = budget.totals()
        if (len(progress["completed"]) != limits["runs"] or budget.updates != limits["total_updates"]
                or totals["charged_seconds"] > limits["gpu_seconds"] or budget.reserved > limits["peak_reserved_bytes"]):
            raise ValueError("screen denominator or resource total differs from the approved scope")
        progress.update(execution="completed", current=None)
    except BaseException as error:
        progress.update(execution="stopped", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(progress_path, progress | budget.totals())


def git(*arguments):
    return subprocess.check_output(["git", *arguments], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path, help="record of separate owner approval")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    screen = read_json(CONFIG)
    revision = git("rev-parse", "HEAD")
    require_authorization(read_json(args.authorization), screen, file_sha(CONFIG), revision)
    git("merge-base", "--is-ancestor", screen["parent"], revision)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("a clean frozen worktree is required")
    execute(args.manifest, args.output, screen, revision)


if __name__ == "__main__":
    main()
