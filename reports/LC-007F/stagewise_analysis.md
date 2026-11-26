# Stagewise results

Labels `early` and `final` are calls 4 and 8. Calls 1–3 and 5–7 are `not_run` in epochs 1–35. The archive verified both readouts in those epochs and all eight readouts in epochs 36–40, within a single existing sweep per policy. [Corrections](stagewise_correction.csv) · [Indexed per-image and grouped metrics](epoch_index.csv)

| Arm | 7→8 Δ MAE | 7→8 Δ local nAP | 4→8 Δ MAE | 4→8 Δ local nAP | High density 4→8 Δ MAE | High density 4→8 Δ local nAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S8 | 0.0000 | +0.003079 | −2.8275 | +0.012575 | −6.6182 | +0.006738 |
| U8 | −12.5050 | +0.009902 | −0.8375 | +0.008016 | −1.8273 | +0.006005 |
| S8RS | −0.2425 | +0.003006 | −2.8675 | +0.013263 | −6.1636 | +0.006701 |
| U8RS | −11.3575 | +0.011224 | +0.0350 | +0.008523 | −0.0545 | +0.006496 |

The table uses EMA E36–40, with equal seed weights. Within K=8 the unshared arms gain more local nAP in the last call, while the shared arms gain more from the supervised midpoint to final. The large U8 and U8RS last-call decreases in FP (1,187.6 and 1,125.1) accompany increases in FN (447.2 and 418.6). Per-image unmatched cancellation decreases by 240.2 and 252.2. Intermediate readout 7 has no direct auxiliary loss. These changes characterize its output distribution and cannot identify recovery of individual heads.

Archived metrics include MAE/RMSE, signed errors, TP/FP/FN, pooled local nAP by density, and state/update RMS by call and policy. RMS uses FP32 means of squares with the same tile and image weighting as K=4. Live and EMA gate logs record min/max/mean/L2 and delta L2 for all eight calls. Cancellation is twice the lesser of FP and FN per image. [Gate logs](gate_statistics.csv)

Tile/cell/slot identity transitions, matched displacements, score and rank changes are `not_available`. The existing decoder exports tile IDs and positions/scores while discarding cell and slot IDs. The frozen schedule keeps the validated matcher path and budgets neither a separate diagnostic match path nor another sweep. [Tiled inference](../../src/loopcrowd/inference.py) · [Decision rules](protocol.md)
