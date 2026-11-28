# Frozen single-case crop × sharing comparison

Execution source `6a83290a2de0c502df0c5c8abd088f26a3b6b56e`, Git tree `f49846b77e65974702c549fde79b4d1aed171714`, execution config SHA-256 `7d78b00ba75eb87cc6165d2945ecbd1422fd7f338056db52ec388ed5412f1415`. The only new case is `42/U4-shift-768`. [Config](../../configs/lc007k_unshared_native768.json)

Use the frozen B train360/internal-val40 split, 44,856/4,295 GT, cached ResNet-18, and 768×768 native random training crops. Batch 2, BF16, live BN during training, AdamW .0002 and .01 decay, 360-step warmup and cosine to .00002, 40×180 successful updates. The ungated unshared variant uses width 192, six heads, four independent blocks and eight-token windows shifted 0/4/0/4. Use stride-4 two-slot predictions with stride-8 P3, FP32 .99 EMA over unique parameters with live BN buffers, and final loss plus .25 midpoint loss. Training reads two outputs. Evaluation uses fixed 512×512 tiles and overlap 128 with one four-stage sweep each for live and EMA. Preserve strict owned count >.30 with δ=.50 and AP floor >.05, δ=.05:.05:.50, with the original train-only density groups.

The preflight checks exact input hashes, decodes all 400 images, verifies U512 initialization and records common encoder, decoder and first block parity with S768. U768 planned crop size, origins, image order, LR, pixels, GT and slot counts match all 40 actual S768 epochs. The older U512 geometry differs from 768 and reproduces its own archived crop/LR trajectory in the same image order. The runner and the archiver check actual U768 training against the plan and its matched S768 counterpart. [Runtime-checked initialization and crop parity](initialization_audit.json)

## Four cells and fixed decisions

| Seed-42 case | E36–40 EMA MAE | Archived local nAP | RMSE |
| --- | ---: | ---: | ---: |
| S512 | 23.8750 | .305537058 | 31.814928 |
| U512 | 20.2550 | .307774651 | 27.283746 |
| S768 | 19.7000 | .346196325 | 25.967582 |
| U768 | Pending | Pending | Pending |

The dedicated CPU archiver scores final EMA E36–40 with equal epoch weight. It reports both same-crop S−U gaps, both within-model crop changes and their difference-in-differences, plus E40, live, density and stage results, TP/FP/FN, signed error and cancellation. All controls are historical and non-contemporaneous. The interaction is descriptive for one seed.

U768 versus U512 requires MAE ≤19.64735, nAP >.3077746512507217, sample MAE CV ≤10%, absolute relative loss change E31–35 to E36–40 ≤3%, RMSE ≤1.03× and FP/FN ≤1.05×. S768 versus U768 reports the strict two-metric direction and the practical MAE margin of S768 ≤.97×U768 with strictly higher nAP, separately from stability, RMSE, FP and FN guards. The existing S768 absolute loss change 4.1255% fails its stability guard independently of U768. S768's historical crop joint fail is preserved. No threshold, epoch or prefix selection is scheduled.

[Runner](../../scripts/run_lc007k.py) · [Launcher](../../scripts/run_lc007k.sh) · [Archiver](../../scripts/archive_lc007k.py)
