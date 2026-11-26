# Stagewise readout plan

All experimental stage metrics are `not_run`. The decoder reads maps from the same eight-call forward. Labels `early` and `final` are calls 4 and 8. Calls 1–3 and 5–7 are `not_run` in epochs 1–35. All eight are planned in epochs 36–40, with no additional tiled sweep.

Keep MAE/RMSE, signed count error, TP/FP/FN and per-image counts, pooled local nAP by density and epoch, and state/update RMS by call and policy. RMS uses FP32 means of squares and the same tile and image weighting as K=4. Report missing calls explicitly and stop on nonfinite states. Gate scales record min/max/mean/L2 and delta L2 through all calls for live and EMA. Report adjacent changes, especially 7→8, and the supervised 4→8 change. Compute count cancellation per image as twice the lesser of FP and FN. Changes in count error alone cannot identify recovery of specific heads.

Tile/cell/slot identity transitions, matched displacements, score and rank changes are `not_available`. The existing decoder exports tile IDs and positions/scores while discarding cell and slot IDs. The frozen schedule keeps the validated matcher path and budgets neither a separate diagnostic match path nor another sweep. [Tiled inference](../../src/loopcrowd/inference.py) · [Decision rules](protocol.md)
