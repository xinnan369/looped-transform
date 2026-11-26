"""Run the shifted-window queue after separate approval, observing device memory."""

import argparse
from pathlib import Path

from loopcrowd.experiment import file_sha
from scripts.prepare_lc007g import SCREEN, audit, history
from scripts.run_lc007d import git, read_json, require_authorization
from scripts.run_lc007e import DepthBudget, execute


class CrossWindowBudget(DepthBudget):
    label = "LC-007G"
    cleanup_reserve_seconds = 20

    def __init__(self, limits, ledger, device):
        if "peak_reserved_bytes" in limits:
            raise ValueError("LC-007G observes memory without a process quota")
        super().__init__(limits, ledger, device)
        self.longest.update(preflight=30, training=20, validation_live=20, validation_ema=20, checkpoint=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    screen = read_json(SCREEN)
    revision = git("rev-parse", "HEAD")
    require_authorization(read_json(args.authorization), screen, file_sha(SCREEN), revision)
    git("merge-base", "--is-ancestor", screen["parent"], revision)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("a clean frozen worktree is required")
    execute(args.manifest, args.output, screen, revision, audit_inputs=audit, history_tables=history,
            budget_type=CrossWindowBudget,
            projection_keys=("upper_with_20pct_contingency_seconds", "required_device_headroom_bytes_estimate"))


if __name__ == "__main__":
    main()
