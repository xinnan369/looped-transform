# Stagewise readout plan

The experimental metrics are `not_run`. Readout labels `stage1`, `early`, `stage3` and `final` identify calls 1, 2, 3 and 4. Reconcile all 40 image IDs, grouped TP/FP/FN, count MAE/RMSE and signed count error under live and EMA. Pool local nAP within each density group and epoch before averaging across epochs. Report adjacent differences and call 4 minus call 1. Count cancellation per image uses `2 × min(FP,FN)`.

Per call, the same forward records mean-square incoming feature and post-state updates in FP32. Reduce by averaging tiles within images and images within the split, then take square roots. The normalized measure divides update RMS by the incoming-state RMS where defined. Nonfinite state or output detection stops validation and preserves evidence. Gate scale summaries record min/max/mean/L2 and delta L2 for each policy, call and branch.

GT identity transitions and candidate displacement, score and rank changes are `not_available`. The frozen decoder discards the cell and slot indices while materializing candidates. The protocol uses the established metric path and allocates time for one sweep per policy. [Tiled decoder](../../src/loopcrowd/inference.py) · [Frozen protocol](protocol.md)
