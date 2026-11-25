# LC-006 supervision, depth and sharing analysis

All numbers below use the three prespecified seeds and the frozen Part B internal-validation split unless marked as A. Late metrics average E36–40 at strict score >0.30. Local nAP uses all original in-frame tile candidates above 0.05. TP, FP and FN refer to stable, one-to-one local matches at δ=0.50 on the owned count mask. Every validation frame set contains 4,295 true heads.

## Replication and supervision

ShanghaiTech A LC-005 reported 234.14 → 221.23 late final MAE with λ=0.25, a 5.51% reduction and improvements in two seeds. A's early MAE was 388.82 → 219.09 and its late final local nAP changed by −0.00176. The absolute A and B endpoints describe different populations.

| B architecture | Seed | Early MAE off → on | Final MAE off → on | Final local nAP Δ | Final FP Δ | Final FN Δ |
|---|---:|---|---|---:|---:|---:|
| shared_2 | 42 | 32.69 → 26.37 | 26.34 → 25.61 | +0.00089 | +7.0 | -1.2 |
| shared_2 | 123 | 38.60 → 26.11 | 26.79 → 24.80 | -0.00074 | -17.2 | -10.4 |
| shared_2 | 2026 | 43.19 → 29.60 | 27.02 → 26.89 | +0.00059 | +20.4 | -33.0 |
| unshared_2 | 42 | 40.58 → 24.23 | 24.12 → 23.74 | +0.00197 | -42.4 | +13.8 |
| unshared_2 | 123 | 70.86 → 25.30 | 25.39 → 25.62 | +0.00273 | -53.8 | -12.0 |
| unshared_2 | 2026 | 28.94 → 26.63 | 25.43 → 24.73 | +0.00049 | -16.8 | -19.6 |

| Replication condition | Shared observed | Shared | Independent observed | Independent diagnostic |
|---|---|:---:|---|:---:|
| Final mean MAE decreases at least 3% | +3.56% | pass | +1.16% | fail |
| Final MAE improves in at least two seeds | 3/3 | pass | 2/3 | pass |
| Final local nAP difference at least −0.005 | +0.00024 | pass | +0.00173 | pass |
| All six runs stable | 123/shared_2_final_only, 123/shared_2_aux_025, 2026/shared_2_final_only, 2026/shared_2_aux_025 | fail | 123/unshared_2_final_only, 123/unshared_2_aux_025, 2026/unshared_2_final_only, 2026/unshared_2_aux_025 | fail |
| Early mean MAE drops ≥3% in mean and improves in two seeds | +28.31%, 3/3 | pass | +45.74%, 3/3 | pass |

For shared two-call models, three-seed late final MAE was 26.71 → 25.77. The paired difference was -0.95 ± 0.94 heads (sample SD).

For unshared two-call models, three-seed late final MAE was 24.98 → 24.70. The paired difference was -0.29 ± 0.47 heads (sample SD).

The shared aux gain minus the independent aux gain was +0.66 heads and +2.40 percentage points. Gain is a decrease in late MAE. Relative gains divide each MAE change by its own architecture's mean final-only MAE.

Both architectures improved mean counting with auxiliary supervision. The measured objective effect therefore extends to the independent blocks in this study.

Early mean MAE improved with this objective in both architectures. The independent-block result places the first-call improvement beyond weight sharing.

## Repeated computation

| Shared aux seed | Early → final MAE | Early → final RMSE | Early → final local nAP | Final − early FP | Final − early FN |
|---:|---|---|---|---:|---:|
| 42 | 26.37 → 25.61 | 36.01 → 34.75 | 0.28147 → 0.29278 | +49.8 | -97.6 |
| 123 | 26.11 → 24.80 | 34.21 → 32.49 | 0.28620 → 0.29351 | -39.4 | -64.6 |
| 2026 | 29.60 → 26.89 | 40.68 → 36.43 | 0.28374 → 0.29133 | +64.2 | -131.8 |

| Independent baseline | Shared aux mean MAE Δ ± sample SD | Relative MAE reduction | Seeds with lower MAE | Mean local nAP Δ |
|---|---:|---:|---:|---:|
| Shared-1 | +0.34 ± 0.92 | -1.34% | 1/3 | +0.00298 |
| CNN | -2.51 ± 3.22 | +8.89% | 2/3 | +0.01775 |

Within the same Shared-2 aux checkpoints, the final-versus-early mean MAE difference was -1.59 and local nAP difference was +0.00874. The direction was favorable in 3/3 seeds for count and 3/3 for localization. The complete repeated-depth gate failed.

| Depth condition | Result |
|---|:---:|
| Mean final MAE decreases at least 3% versus Shared-1 | fail |
| At least two seed MAEs improve versus Shared-1 | fail |
| Final local nAP delta at least −0.005 versus Shared-1 | pass |
| Within-checkpoint count or localization improves consistently | pass |

## Weight sharing with the same auxiliary objective

| Seed | Shared minus independent final MAE | Final RMSE Δ | Final local nAP Δ |
|---:|---:|---:|---:|
| 42 | +1.87 | +2.86 | -0.00596 |
| 123 | -0.82 | -2.22 | -0.00369 |
| 2026 | +2.16 | +2.51 | -0.00214 |
| Mean ± sample SD | +1.07 ± 1.64 | +1.05 ± 2.84 | -0.00393 ± 0.00192 |

The shared arm uses 1,596,614 parameters and the independent arm uses 2,041,478. Sharing saves 21.79% of the independent parameter count. Both perform the same two fixed-window calls and use the same loss weighting.

| Architecture | Parameter count | Default forward GFLOP/tile | Profile train peak allocated MiB | Profile step ms | Batch-one tile ms | Full-image ms |
|---|---:|---:|---:|---:|---:|---:|
| cnn_only | 1,151,750 | 25.095 | 358.7 | 29.35 | 6.13 | 73.33 |
| shared_1 | 1,596,614 | 28.920 | 439.5 | 35.03 | 8.40 | 87.17 |
| shared_2 | 1,596,614 | 32.745 | 585.3 | 64.67 | 10.88 | 100.96 |
| unshared_2 | 2,041,478 | 32.745 | 591.2 | 56.62 | 10.25 | 96.94 |

The paired prediction metrics and lower parameter count describe a parameter–accuracy trade-off.

Profile measurements use a resident synthetic BF16 training batch of two 512 px tiles, then batch-one inference on the same held-out frame, with nine synchronized measurements after warm-up. Two-call training profiles use λ=0.25. Inference uses one final decode. FLOPs count two operations per MAC in convolution, linear and attention matrix multiplication. The preflight includes raw latency samples, GPU telemetry before each profile, allocator reserves and seed-specific warm-up memory. Training timings include input and validation costs. Local CPU testing overlapped the first run and workstation activity can vary over the queue. The timing observations are descriptive.

Paired means, sample SDs, best and E40 metrics and all seed-specific observations are in [paired seed results](paired_seed_results.csv). The 18 run-level stability figures and measured peak reserved/allocated memory are in the [experiment matrix](experiment_matrix.csv).

Interpret these three seeds as internal-validation pilots. Local nAP uses the stated spacing-relative protocol.
