# Stagewise results

Readout labels `stage1`, `early`, `stage3` and `final` identify calls 1, 2, 3 and 4. All 40 image IDs and grouped count metrics reconcile under live and EMA. Local nAP is pooled within each density group and epoch before averaging. This table reports call 4 minus call 1 on EMA E36–40, equally weighted across seeds. Count cancellation per image uses `2 × min(FP,FN)`. [Adjacent and first-to-final changes](stagewise_correction.csv) · [Per-image inputs](epoch_index.csv)

| Arm | All Δ MAE | All Δ local nAP | High-density Δ MAE | High-density Δ local nAP | All Δ FP | All Δ FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S4 | −32.2000 | +0.034781 | −75.7273 | +0.027707 | +776.7 | −1,039.3 |
| U4 | −19.8575 | +0.052791 | −44.0091 | +0.038257 | +387.1 | −878.0 |
| S4RS | −35.9525 | +0.034572 | −84.8818 | +0.028457 | +878.0 | −1,149.3 |
| U4RS | −19.7900 | +0.052634 | −41.9182 | +0.038329 | +341.4 | −841.4 |

The shared arms start with larger count errors and reduce them more by call 4. The unshared arms gain more local nAP in both density views and finish with lower MAE. The all-image cancellation terms increase by 1,025.4 for S4 and 1,166.8 for S4RS, versus 303.4 for U4 and 291.6 for U4RS. The aggregate changes include both increased false positives and reduced false negatives.

Per call, the same forward records mean-square incoming feature and post-state updates in FP32. Reduce by averaging tiles within images and images within the split, then take square roots. The normalized measure divides update RMS by the incoming-state RMS where defined. Nonfinite state or output detection stops validation and preserves evidence. Gate scale summaries record min/max/mean/L2 and delta L2 for each policy, call and branch.

GT identity transitions and candidate displacement, score and rank changes are `not_available`. The frozen decoder discards the cell and slot indices while materializing candidates. The protocol uses the established metric path and allocates time for one sweep per policy. [Tiled decoder](../../src/loopcrowd/inference.py) · [Frozen protocol](protocol.md)
