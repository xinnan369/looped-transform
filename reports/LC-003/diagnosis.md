# LC-003 historical instability diagnosis

The LC-002R archive provides 140 epoch records across seven seed-42 variants. Each run processed 270 shuffled training crops per epoch and evaluated the same 30 internal validation images (14,177 heads). The historical schedule was constant AdamW at 0.0002, batch two, BF16, random native-scale 512 px crops and two point slots per stride-four cell. The configured budget was 135 batches per epoch and 2,700 updates over 20 epochs. The archive contains epoch batch coverage and final-step gradient norms. Its optimizer-update counts are inferred from batch coverage and BF16 execution.

The manifest and ImageNet1K ResNet18 cache digests match the archived runs. Both checkpoint files for all seven variants match their archived digests. The common projection, fuse and point heads were initialized with seed 42 before creating any transformer blocks. The classification bias started at −4.59511985 (sigmoid 0.01). Backbone weights came from the existing local cache. See [the structured diagnosis](diagnosis.json) for exact provenance, the 140 epoch records and the seven fresh validation sweeps.

## Archived epoch results

| Variant | Best epoch | MAE best → final | RMSE best → final | Local nAP best → final | Predicted heads best → final | Last-five mean MAE | CV | Loss change E11–15 to E16–20 | Median final-step grad L2 E1–5 → E16–20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cnn_only | 15 | 197.47 → 241.23 | 262.92 → 396.00 | 0.1597 → 0.1549 | 13,899 → 18,292 | 251.48 | 9.86% | -1.69% | 0.588 → 0.407 |
| shared_1 | 19 | 162.63 → 281.20 | 259.50 → 486.10 | 0.1917 → 0.1396 | 14,522 → 19,627 | 248.51 | 23.05% | -2.24% | 0.748 → 0.657 |
| shared_2 | 19 | 181.90 → 266.73 | 272.29 → 434.81 | 0.1793 → 0.1435 | 12,934 → 17,807 | 255.38 | 24.94% | -2.45% | 1.367 → 0.497 |
| shared_4 | 17 | 192.50 → 269.83 | 300.54 → 432.35 | 0.1816 → 0.1427 | 12,318 → 17,202 | 266.48 | 28.66% | -2.76% | 1.582 → 0.451 |
| unshared_4 | 17 | 171.23 → 239.57 | 284.80 → 378.50 | 0.1896 → 0.1469 | 12,472 → 15,808 | 240.29 | 21.78% | -2.62% | 1.023 → 0.432 |
| shifted_shared_2 | 19 | 176.27 → 286.20 | 253.92 → 470.09 | 0.1873 → 0.1305 | 13,911 → 17,263 | 277.53 | 25.82% | -2.35% | 1.350 → 0.525 |
| shifted_shared_4 | 19 | 208.73 → 286.93 | 298.68 → 490.49 | 0.1757 → 0.1275 | 10,685 → 16,949 | 301.28 | 26.82% | -2.59% | 1.317 → 0.733 |

The final-step gradients were all finite. Their median declined in every variant across these two windows. Shared-1 had the lowest best MAE and a 23.05% late CV. Its late mean MAE was 2.97 heads below CNN. The epoch records expose count swings against a fixed target total even while mean training loss falls. The gradient samples cover one step per epoch.

## Error by image density

Density is GT heads per megapixel of the original image. Inclusive linear tertiles from the 270 training labels fixed the boundaries at 545.60 and 1080.58 before inspecting validation errors. Low includes the first boundary and mid includes the second. Archived per-image predictions are unavailable. Re-evaluation of the seven historical best checkpoints reproduced their aggregate fixed-threshold count, MAE, RMSE and local nAP. The table summarizes the re-evaluated validation images. Signed bias is mean predicted minus true heads per image.

| Variant | Group | Images | MAE | RMSE | Signed bias |
|---|---|---:|---:|---:|---:|
| cnn_only | low | 11 | 99.00 | 122.05 | +19.73 |
| cnn_only | mid | 8 | 267.00 | 342.36 | -56.50 |
| cnn_only | high | 11 | 245.36 | 297.30 | -3.91 |
| shared_1 | low | 11 | 76.36 | 126.31 | +8.73 |
| shared_1 | mid | 8 | 220.88 | 296.53 | -11.38 |
| shared_1 | high | 11 | 206.55 | 322.11 | +30.91 |
| shared_2 | low | 11 | 86.91 | 137.52 | +12.18 |
| shared_2 | mid | 8 | 198.38 | 269.41 | -61.12 |
| shared_2 | high | 11 | 264.91 | 361.26 | -80.73 |
| shared_4 | low | 11 | 105.18 | 144.46 | +12.82 |
| shared_4 | mid | 8 | 224.75 | 321.79 | -80.75 |
| shared_4 | high | 11 | 256.36 | 387.52 | -123.09 |
| unshared_4 | low | 11 | 86.64 | 122.48 | -1.00 |
| unshared_4 | mid | 8 | 201.62 | 302.70 | -46.88 |
| unshared_4 | high | 11 | 233.73 | 373.59 | -119.91 |
| shifted_shared_2 | low | 11 | 85.00 | 131.23 | -14.27 |
| shifted_shared_2 | mid | 8 | 215.38 | 287.83 | -50.12 |
| shifted_shared_2 | high | 11 | 239.09 | 313.63 | +26.55 |
| shifted_shared_4 | low | 11 | 85.27 | 100.33 | -60.73 |
| shifted_shared_4 | mid | 8 | 280.62 | 367.50 | -144.88 |
| shifted_shared_4 | high | 11 | 279.91 | 367.44 | -151.36 |

The low-density group has lower MAE for all seven checkpoints. The mid- and high-density groups show larger absolute errors and often undercount. Small aggregate count bias can hide offsetting image-level overcounts and undercounts.

## Score sensitivity and localization

Count metrics use an ownership filter and a strict score >0.30 cutoff. Local nAP ranks in-frame tile candidates at its frozen 0.05 score floor and averages ten spacing-relative matching radii. Changing the count cutoff in a frozen checkpoint measures count calibration. Changes in local nAP across checkpoints can reflect localization and score ordering, so they support a separate diagnostic.

| Variant | Best checkpoint MAE at 0.25 | At 0.30 | At 0.35 | Local nAP change best → final |
|---|---:|---:|---:|---:|
| cnn_only | 621.37 | 197.47 | 317.17 | -0.0048 |
| shared_1 | 682.63 | 162.63 | 318.27 | -0.0521 |
| shared_2 | 607.37 | 181.90 | 344.90 | -0.0358 |
| shared_4 | 624.33 | 192.50 | 357.93 | -0.0389 |
| unshared_4 | 649.80 | 171.23 | 362.27 | -0.0427 |
| shifted_shared_2 | 664.90 | 176.27 | 333.93 | -0.0568 |
| shifted_shared_4 | 503.93 | 208.73 | 381.60 | -0.0482 |

All seven best checkpoints prefer 0.30 among the tested cutoffs. The adjacent cutoff results quantify sensitivity at those checkpoints. The epoch-history nAP uses the same ranking protocol throughout. Per-epoch score distributions and coordinates were unavailable, which limits attribution of the MAE swings to score or geometry.

## Hypotheses

- Batch-two backbone batch normalization could move running statistics as heterogeneous crops arrive. Measure checkpoint BN statistics and compare frozen versus train-mode BN with the other controls fixed.
- Constant LR could keep predictions moving late in training. The preregistered 40-epoch comparison tests this by varying the per-update schedule with identical crops and initial states.
- Native-scale random crops change the density mix over epochs. Compare archived sampled head totals, then fix the crop sequence in a separate diagnostic if residual variation warrants it.
- Focal weights and positive-slot normalization can make sparse and crowded batches contribute different gradients. The new runs record realized positive and negative slots, overflow, classification loss and offsets so those relationships can be measured.

These four hypotheses remain open in the historical evidence.
