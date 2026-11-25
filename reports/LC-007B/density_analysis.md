# LC-007B density and paired error analysis

The B train-only density tertiles are 82.6518 and 180.9862 heads/Mpx. The fixed validation groups contain
13 low-density, 16 mid-density and 11 high-density images. The comparison uses the independently trained
models within a seed and the same live or EMA policy. Local nAP is pooled inside each stratum for each
epoch before averaging E36–40. Counts and matched TP/FP/FN use the strict owned count mask.

Shared-2 aux's own EMA-minus-live E36–40 changes show whether a narrowed comparator gap
coincided with an absolute regression in another stratum.

| Stratum | Seeds | ΔMAE | ΔRMSE | Δ local nAP | ΔFP | ΔFN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 2 | -3.27 | -4.34 | +0.00217 | +27.8 | -55.7 |
| low | 2 | -1.32 | -1.19 | +0.00200 | +9.5 | -10.8 |
| mid | 2 | -1.44 | -2.29 | +0.00130 | +7.9 | -22.3 |
| high | 2 | -8.24 | -7.89 | +0.00267 | +10.4 | -22.6 |

Each entry is the Shared-2 aux value minus its matched comparator value. Negative MAE gaps favor
Shared-2. Gap change is EMA gap minus live gap. The macro rows in the CSV average the available
completed seeds with equal weight. Rows with one seed describe that seed only.

| Comparator | Stratum | Seeds | MAE gap live → EMA (change) | RMSE gap live → EMA | nAP gap live → EMA |
| --- | --- | ---: | ---: | ---: | ---: |
| cnn_only | all | 2 | -4.03 → -5.28 (-1.24) | -5.80 → -7.38 | +0.01853 → +0.01982 |
| cnn_only | low | 2 | -2.28 → -2.73 (-0.45) | -2.16 → -2.55 | +0.02353 → +0.02428 |
| cnn_only | mid | 2 | -0.67 → -1.61 (-0.94) | +0.18 → -1.21 | +0.02141 → +0.02276 |
| cnn_only | high | 2 | -10.99 → -13.62 (-2.63) | -13.01 → -15.39 | +0.01471 → +0.01581 |
| shared_1 | all | 2 | -0.17 → -0.58 (-0.41) | +0.15 → -0.39 | +0.00413 → +0.00492 |
| shared_1 | low | 2 | -0.63 → -0.87 (-0.24) | -0.39 → -0.39 | -0.00199 → -0.00122 |
| shared_1 | mid | 2 | +0.79 → +0.60 (-0.19) | +1.32 → +0.92 | +0.00739 → +0.00818 |
| shared_1 | high | 2 | -1.00 → -1.95 (-0.95) | -0.56 → -1.75 | +0.00320 → +0.00400 |
| unshared_2_aux_025 | all | 2 | +0.53 → +1.08 (+0.55) | +0.32 → +1.62 | -0.00483 → -0.00515 |
| unshared_2_aux_025 | low | 2 | +0.06 → -0.37 (-0.43) | +0.25 → +0.50 | -0.00876 → -0.01109 |
| unshared_2_aux_025 | mid | 2 | +0.85 → +1.78 (+0.93) | +1.10 → +2.56 | -0.00272 → -0.00332 |
| unshared_2_aux_025 | high | 2 | +0.60 → +1.77 (+1.17) | -0.15 → +1.35 | -0.00459 → -0.00422 |

| Comparator | Stratum | Signed count gap live → EMA | ΔTP live → EMA | ΔFP live → EMA | ΔFN live → EMA |
| --- | --- | ---: | ---: | ---: | ---: |
| cnn_only | all | +460.7 → +408.8 | +313.6 → +312.9 | +147.1 → +95.9 | -313.6 → -312.9 |
| cnn_only | low | +65.3 → +69.1 | +46.2 → +50.1 | +19.1 → +19.0 | -46.2 → -50.1 |
| cnn_only | mid | +121.3 → +108.9 | +102.5 → +105.6 | +18.8 → +3.3 | -102.5 → -105.6 |
| cnn_only | high | +274.1 → +230.8 | +164.9 → +157.2 | +109.2 → +73.6 | -164.9 → -157.2 |
| shared_1 | all | +87.2 → +39.9 | +58.6 → +48.0 | +28.6 → -8.1 | -58.6 → -48.0 |
| shared_1 | low | +12.8 → +12.5 | +7.7 → +8.6 | +5.1 → +3.9 | -7.7 → -8.6 |
| shared_1 | mid | +28.2 → +14.2 | +22.3 → +20.0 | +5.9 → -5.8 | -22.3 → -20.0 |
| shared_1 | high | +46.2 → +13.2 | +28.6 → +19.4 | +17.6 → -6.2 | -28.6 → -19.4 |
| unshared_2_aux_025 | all | -11.2 → -47.8 | -46.8 → -63.7 | +35.6 → +15.9 | +46.8 → +63.7 |
| unshared_2_aux_025 | low | -22.4 → -22.2 | -16.9 → -17.4 | -5.5 → -4.8 | +16.9 → +17.4 |
| unshared_2_aux_025 | mid | +9.2 → -7.7 | -12.5 → -17.2 | +21.7 → +9.5 | +12.5 → +17.2 |
| unshared_2_aux_025 | high | +2.0 → -17.9 | -17.4 → -29.1 | +19.4 → +11.2 | +17.4 → +29.1 |

- **cnn_only.** The high-density MAE gap moved toward Shared-2 (-2.63 heads). The low- and mid-density gap changes were -0.45 and -0.94.
- **shared_1.** The high-density MAE gap moved toward Shared-2 (-0.95 heads). The low- and mid-density gap changes were -0.24 and -0.19.
- **unshared_2_aux_025.** The high-density MAE gap moved toward the comparator (+1.17 heads). The low- and mid-density gap changes were -0.43 and +0.93.

The [architecture table](architecture_comparison.csv) retains both seeds, secondary E40
rows and paired fractions of images with better, unchanged and worse late absolute error. The
[image pairs](architecture_image_pairs.csv) retain the joined IDs, both policies' per-image late
MAE and signed gaps, and the EMA-minus-live change. Repeated readings of the same validation
images and the two training seeds are descriptive development evidence.
