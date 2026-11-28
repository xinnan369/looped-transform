"""Archive one LoopNorm run and apply the independently frozen S768/U768 gates."""

import argparse
from math import isfinite
from pathlib import Path

from scripts.archive_lc007d import endpoint, stability
from scripts.archive_lc007i import archive as archive_native
from scripts.prepare_lc007k import SHARED_ARM, UNSHARED_ARM, assert_paired_exposure
from scripts.prepare_lc007l import CASE, SCREEN, controls
from scripts.run_lc007a import write_csv
from scripts.run_lc007d import read_json
from scripts.run_lc007l import authorize


def score(training, metrics, rule, *, s768, u768):
    def primary(table):
        return endpoint(table, rule["analysis_epochs"], "ema", "final", "all")
    candidate = primary(metrics)
    stable = stability(training, metrics, rule)
    references = {}
    for label, data, max_mae in ((SHARED_ARM, s768, rule["s768_mae_max"]),
                                 (UNSHARED_ARM, u768, rule["u768_mae_max"])):
        base = primary(data["metrics"])
        raw = {"mae": candidate["mae"] < base["mae"], "nap": candidate["local_nap"] > base["local_nap"]}
        practical = {"mae": candidate["mae"] <= max_mae, "nap": raw["nap"]}
        guards = {"stability": stable["passed"]} | {
            key: candidate[key] <= rule[f"{key}_ratio"] * base[key] for key in ("rmse", "fp", "fn")}
        references[label] = {"reference": base, "raw": raw, "practical": practical, "guards": guards,
                             "joint_pass": all(practical.values()) and all(guards.values()),
                             "historical_crop_joint_gate": "fail"}
    return {"candidate": candidate, "stability": stable, "comparisons": references,
            "decision": "pass" if references[UNSHARED_ARM]["joint_pass"] else "fail"}


def diagnostic_rows(history):
    for epoch in history:
        grads = epoch["loop_norm_gradient_l2_mean"]
        policies = epoch["loop_norm_affine_l2"]
        if len(grads) != 4 or set(policies) != {"live", "ema"}:
            raise ValueError("normalization diagnostic coverage differs")
        for policy, affine in policies.items():
            for loop, (norms, grad_norms) in enumerate(zip(affine, grads, strict=True), 1):
                if len(norms) != 2 or len(grad_norms) != 2:
                    raise ValueError("normalization branch coverage differs")
                for branch, a, g in zip(("norm1", "norm2"), norms, grad_norms, strict=True):
                    if (len(a) != 4 or len(g) != 2 or any(not isfinite(value) or value < 0 for value in (*a, *g))
                            or (loop == 1 and any(a[2:]))):
                        raise ValueError("invalid normalization diagnostics")
                    yield {"epoch": epoch["epoch"], "policy": policy, "loop": loop, "branch": branch,
                           "gamma_shift_l2": a[0], "beta_shift_l2": a[1],
                           "gamma_vs_loop1_l2": a[2], "beta_vs_loop1_l2": a[3],
                           "live_mean_gamma_gradient_l2": g[0], "live_mean_beta_gradient_l2": g[1]}


def archive(root, output, plan):
    (shared, _), (unshared, _) = controls(plan)
    def evaluate(training, metrics, _historical, rule):
        return score(training, metrics, rule, s768=shared, u768=unshared)
    def verify(training):
        for control in (shared, unshared):
            assert_paired_exposure(training, control["training"][:len(training)])
        history = read_json(root / "run" / "train" / CASE / "history.json")
        if len(history) != len(training):
            raise ValueError("normalization history length differs")
        write_csv(output / "loop_norm_diagnostics.csv", list(diagnostic_rows(history)))
    archive_native(root, output, plan, screen=SCREEN, authorize=authorize, scorer=evaluate,
                   comparisons={SHARED_ARM: shared["metrics"], UNSHARED_ARM: unshared["metrics"]},
                   verify_exposure=verify)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive(args.run_root, args.output, read_json(SCREEN))


if __name__ == "__main__":
    main()
