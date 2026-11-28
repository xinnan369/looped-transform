# Frozen native-768 loop-specific normalization

Execution commit `17eb61fda1e94b73fca9bc539f0c4499622950da`, Git tree `2bd3ff2c6f25cda0033abc2cc848ab0514df5a30`, execution config SHA-256 `4ab8b7622ebc50f9341ef4087d3821d62d0ec2e59e82a8d9e21fa6090d551f3c`. The sole new case is `42/S4-shift-768-LN`. [Configuration](../../configs/lc007l_loopnorm.json)

Loop-specific affine parameters may let stages specialize while attention and the FFN share weights. The motivation follows MiniViT's layer-specific normalization under weight sharing. MiniViT also uses weight transformations and distillation. This single-factor experiment changes the two LayerNorm affine pairs per call. [MiniViT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.pdf)

The four-call S4-shift block retains its normalization axes, eps, order, attention, FFN, width 192, six heads and noncyclic window-8 shifts 0/4/0/4. Call 1 uses the two original LayerNorms. Calls 2–4 each use exact copied affine initial states. No refinement mode, distillation or auxiliary losses are enabled.

## Fixed execution recipe

Use cached B train360 and internal-val40, GT 44,856/4,295, native 1024×768 frames, native random 768×768 training crops and evaluation 512×512 tiles with overlap 128. Retain batch 2 BF16, train BN, two stride-4 slots, P3 stride 8, AdamW .0002, weight decay .01, warmup 360 and cosine to .00002, FP32 EMA .99 over unique parameters with live BN buffers. Forty epochs of 180 successful updates yield 7,200 total. Training uses final loss plus .25 midpoint loss with two supervised decodes. Live and EMA each use one four-stage sweep every epoch. Count remains owned score >.30, δ=.50. Local nAP uses score >.05 and δ=.05:.05:.50 with frozen train-only density strata.

Actual image-order, crop and LR hashes, pixels, GT, overflow and slot exposures are verified at runtime and archival against the S768 and U768 controls. The 40 planned epochs include 8,493,465,600 valid crop pixels, 1,395,609 sampled GT and 180 over-slot GT. [Matched exposure and source hashes](initialization_audit.json)

## Frozen decision

The primary readout is E36–40 final EMA, equal epoch weights. Historical controls share the seed and split and were run earlier.

| Reference | MAE | nAP | RMSE | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S768 | 19.7000 | .346196325464257 | 25.96758 | 3,066.8 | 1,760.6 | 1,228.2 |
| U768 | 19.1600 | .351293878523962 | 25.97379 | 3,070.2 | 1,730.0 | 1,224.8 |

Raw primary superiority requires MAE <19.16 and nAP above the exact archived U768 value. Practical U768 superiority requires MAE ≤18.5852 and strictly higher nAP. Report S768 improvement separately at MAE ≤19.109 and strictly higher nAP. Against each reference, the full gate also requires sample MAE CV ≤10%, absolute relative change in mean final loss from E31–35 to E36–40 ≤3%, RMSE ≤1.03× and FP/FN ≤1.05×. Both historical crop failures are preserved. Report secondary E40 and live, density and stagewise results, TP/FP/FN, signed error and cancellation. A single modified model against historical controls leaves sharing specificity and cross-seed generalization open.

## Diagnostics and archival schema

Each epoch records the mean per-batch FP32 L2 gradient of gamma and beta after unscale and before update. For each loop and norm branch, live and EMA snapshots record gamma's L2 shift from one, beta's L2 shift from zero, and the corresponding pairwise L2 distances to the same branch at call 1. Validation takes these snapshots during the existing passes. The archiver writes 640 expected norm rows for a complete run, indexed by epoch, policy, loop and norm branch. Gradient values are training measurements repeated alongside the two weight policies. Existing per-stage state/update RMS remains in standard metrics. Affine divergence and accuracy changes are descriptive associations.

[Runner](../../scripts/run_lc007l.py) · [Launch deadline](../../scripts/run_lc007l.sh) · [Archiver and fixed decision](../../scripts/archive_lc007l.py)
