"""Guard the one-case unshared native-crop execution and CPU precheck."""

import argparse
import signal
from pathlib import Path

from loopcrowd.experiment import file_sha, write_json
from scripts.prepare_lc007k import SCREEN, audit
from scripts.run_lc007d import git, read_json
from scripts.run_lc007h_e40 import stop_at_deadline
from scripts.run_lc007h_training import require_training_authorization
from scripts.run_lc007i import execute


def authorize(owner, plan, digest, revision, tree):
    require_training_authorization(owner, plan, digest, revision, tree)
    review = owner.get("independent_review")
    if not isinstance(review, str) or not review.strip() or owner.get("owner_review_waiver"):
        raise ValueError("LC-007K requires its own independent review record and separate launch approval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--authorization", type=Path)
    mode.add_argument("--cpu-verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(SCREEN)
    if args.cpu_verify_only:
        write_json(args.output, audit(args.manifest, plan))
        return
    revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    authorize(read_json(args.authorization), plan, file_sha(SCREEN), revision, tree)
    git("merge-base", "--is-ancestor", plan["parent"], revision)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("the frozen checkout must be clean")
    signal.signal(signal.SIGTERM, stop_at_deadline)
    execute(args.manifest, args.output, plan, revision, tree, screen=SCREEN, preflight=audit)


if __name__ == "__main__":
    main()
