"""Summarize the four cells while preserving the original complete/partial audit."""

import argparse
from pathlib import Path

from scripts.archive_lc007d import endpoint, stability
from scripts.archive_lc007i import archive as native_archive, score as crop_score
from scripts.prepare_lc007i import reference_tables
from scripts.prepare_lc007k import SCREEN, SHARED_512, SHARED_ARM, UNSHARED_ARM, assert_paired_exposure, shared_control
from scripts.run_lc007d import read_json
from scripts.run_lc007k import authorize


def score(training, metrics, historical, rule, *, shared512, shared768):
    result = crop_score(training, metrics, historical, rule)
    def primary(rows):
        return endpoint(rows, rule["analysis_epochs"], "ema", "final", "all")
    s512, s768 = primary(shared512["metrics"]), primary(shared768["metrics"])
    u512, u768 = result["historical_512"], result["candidate"]
    gap512 = {key: s512[key] - u512[key] for key in s512}
    gap768 = {key: s768[key] - u768[key] for key in s768}
    crop_shared = {key: s768[key] - s512[key] for key in s512}
    crop_unshared = {key: u768[key] - u512[key] for key in u512}
    stable = stability(shared768["training"], shared768["metrics"], rule)
    raw = {"mae": s768["mae"] < u768["mae"], "nap": s768["local_nap"] > u768["local_nap"]}
    margins = {"mae": s768["mae"] <= rule["sharing_mae_ratio"] * u768["mae"], "nap": raw["nap"]}
    guards = {"stability": stable["passed"]}
    for key in ("rmse", "fp", "fn"):
        guards[key] = s768[key] <= rule[f"{key}_ratio"] * u768[key]
    return result | {
        "reference_shared768_original_crop_gate": "fail",
        "grid": {SHARED_512: s512, "U4-shift": u512, SHARED_ARM: s768, UNSHARED_ARM: u768},
        "shared_minus_unshared_512": gap512, "shared_minus_unshared_768": gap768,
        "crop_effect_shared": crop_shared, "crop_effect_unshared": crop_unshared,
        "difference_in_differences_shared_minus_unshared": {key: gap768[key] - gap512[key] for key in gap768},
        "sharing": {"raw": raw, "practical_margin": margins, "guards": guards,
                    "shared_stability": stable, "joint_pass": all(margins.values()) and all(guards.values())},
    }


def archive(root, output, plan):
    shared512 = reference_tables(plan | {"historical_reference": SHARED_512})
    shared768, _ = shared_control(plan)
    def compare(epochs, metrics, historical, rule):
        return score(epochs, metrics, historical, rule, shared512=shared512, shared768=shared768)
    def verify_exposure(training):
        assert_paired_exposure(training, shared768["training"][:len(training)])
    native_archive(root, output, plan, screen=SCREEN, authorize=authorize, scorer=compare,
                   comparisons={SHARED_512: shared512["metrics"], SHARED_ARM: shared768["metrics"]},
                   verify_exposure=verify_exposure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.run_root, args.output, read_json(SCREEN))


if __name__ == "__main__":
    main()
