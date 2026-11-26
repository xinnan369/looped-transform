"""Run the reviewed K=8 factorial only with separately recorded owner approval."""

import argparse
from pathlib import Path

from loopcrowd.experiment import file_sha
from scripts.prepare_lc007f import SCREEN, audit, history
from scripts.run_lc007d import git, read_json, require_authorization
from scripts.run_lc007e import DepthBudget, execute


class DepthEightBudget(DepthBudget):
    label = "LC-007F"
    cleanup_reserve_seconds = 20

    def __init__(self, limits, ledger, device):
        super().__init__(limits, ledger, device)
        self.longest.update(preflight=30, training=20, validation_live=25, validation_ema=25, checkpoint=2)


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
    execute(args.manifest, args.output, screen, revision, audit_inputs=audit, history_tables=history,
            budget_type=DepthEightBudget,
            projection_keys=("upper_with_20pct_contingency_seconds", "peak_reserved_projection_bytes"),
            latency_modes=("final_only", "midpoint_final", "all_readouts"), free_disk_gib=4)


if __name__ == "__main__":
    main()
