"""Score the approved eight-call queue from verified local artifacts."""

import argparse
from pathlib import Path

from scripts.archive_lc007c import SHARED, UNSHARED
from scripts.archive_lc007e import archive
from scripts.prepare_lc007f import SCREEN, history

JOINT = (
    ("S8", "S4", "depth4_ungated"), ("S8RS", "S4RS", "depth4_RS"),
    ("S8", SHARED, "depth2_ungated"), ("S8RS", "RS", "depth2_RS"),
    ("S8", "U8", "sharing_ungated"), ("S8RS", "U8RS", "sharing_RS"),
    ("S8RS", "S8", "gating_shared"), ("U8RS", "U8", "gating_unshared"),
)
CONTRASTS = JOINT + (
    *((arm, reference, "historical_target") for arm in ("S8", "S8RS")
      for reference in ("cnn_only", "shared_1", UNSHARED)),
    *((arm, "R", "historical_descriptive") for arm in ("S8", "S8RS")),
)
INTERACTIONS = (
    ("S8", "U8", "S4", "U4", "K4_ungated"), ("S8RS", "U8RS", "S4RS", "U4RS", "K4_RS"),
    ("S8", "U8", SHARED, UNSHARED, "K2_ungated"), ("S8RS", "U8RS", "RS", "U-RS", "K2_RS"),
)
SELECTION = (("S8", ("S4", SHARED, "U8")), ("S8RS", ("S4RS", "RS", "U8RS")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive(args.run_root, args.output, screen_path=SCREEN, history_tables=history,
            contrasts=CONTRASTS, joint=JOINT, strata=INTERACTIONS, selection=SELECTION,
            negative_protocol="retain_K2_and_review_absolute_MAE_trend", depth_calls=8)


if __name__ == "__main__":
    main()
