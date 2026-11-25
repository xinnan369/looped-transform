# LC-005 early and final diagnostics

Both training objectives include the same two decoder paths. Each stage prediction shares fuse, classification and offset parameters. Early scores are diagnostic. The primary comparison uses late final outputs and each run's final-MAE selected best snapshot.

## Paired effects by seed

| Seed | Early last-five MAE Δ | Early local nAP Δ | Final last-five MAE Δ | Final local nAP Δ | Best final MAE Δ | E40 final MAE Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | -175.63 | +0.03379 | +3.65 | +0.00194 | -12.73 | +14.23 |
| 123 | -201.99 | +0.04835 | -16.59 | -0.00192 | -2.00 | -29.57 |
| 2026 | -131.57 | +0.02069 | -25.77 | -0.00529 | -14.87 | -29.67 |

Deltas are auxiliary minus final-only. Each last-five metric averages epochs 36–40 within a seed. See the [paired CSV](paired_seed_results.csv) for means and sample SDs.

## Where count differences arose

| Seed | Stage | Predicted-head Δ | Local TP Δ | Local FP Δ | Local FN Δ |
|---:|---|---:|---:|---:|---:|
| 42 | early | -7068.4 | -872.2 | -6196.2 | +872.2 |
| 123 | early | -6401.8 | -390.2 | -6011.6 | +390.2 |
| 2026 | early | -6845.6 | -1191.6 | -5654.0 | +1191.6 |
| 42 | final | +392.2 | +134.2 | +258.0 | -134.2 |
| 123 | final | -644.2 | -123.0 | -521.2 | +123.0 |
| 2026 | final | -863.4 | -236.0 | -627.4 | +236.0 |

Counts above average the five epochs. Local TP, FP and FN use owned score >0.30 predictions matched greedily at spacing-relative δ=0.50 with stable ties. GT is 14,177 heads at every epoch. Local nAP pools the in-frame candidate ranks above 0.05.

At the first call, mean MAE changed by -169.73 and mean local nAP by +0.03428. Local FP changed by -5953.9 and local FN by +818.0. At the final call, mean MAE changed by -12.90 and local nAP by -0.00176. Final FP and FN changed by -296.9 and +74.9, respectively. The complete gate incorporates stability and consistency across seeds.

Early supervision improved first-call count error and ranked localization across the three paired seeds. The smaller final count improvement coincided with fewer local false positives and more false negatives at the fixed cutoff. Final local nAP fell within the allowed tolerance, with one positive and two negative seed-level changes. These results support a preliminary final counting benefit and leave final localization improvement unestablished.

## Timing and VRAM

| Seed | Arm | Warm-up batch peak allocated MiB | Warm batch median ms | Train batch wall ms | Peak training allocated MiB | Epoch train + val seconds |
|---:|---|---:|---:|---:|---:|---:|
| 42 | final_only | 563.8 | 52.82 | 26.95 | 587.0 | 353.0 |
| 42 | aux_025 | 572.8 | 51.32 | 30.91 | 587.0 | 377.6 |
| 123 | final_only | 572.8 | 17.17 | 29.99 | 587.0 | 386.2 |
| 123 | aux_025 | 572.8 | 51.42 | 25.67 | 587.0 | 333.8 |
| 2026 | final_only | 572.8 | 18.84 | 27.53 | 587.0 | 362.1 |
| 2026 | aux_025 | 572.8 | 17.28 | 30.51 | 587.0 | 377.4 |

GPU allocator peaks describe this process's allocations. The synchronized batch medians use disposable preflight batches and include forward, backward and AdamW. Training batch times divide training wall time, including input loading, by attempted updates. Epoch duration includes both validation stages. The A/B order alternates across seeds to partially balance workstation scheduling and thermal variation. The local CPU test matrix overlapped the first training arm. Recorded epoch and batch timings include system and background workload variation.

## Historical context

| Seed | Historical LC-003 Shared-2 last-five MAE | LC-005 matched control last-five MAE | Epochs with exact final MAE, RMSE and local nAP agreement |
|---:|---:|---:|---:|
| 42 | 218.77 | 218.77 | 40/40 |
| 123 | 230.53 | 230.53 | 40/40 |
| 2026 | 253.10 | 253.10 | 40/40 |

LC-003 uses the same data, weights and schedule with the historical single-decode training graph. The LC-005 control has both decodes and the zero-weight early loss in the graph. The LC-005 A/B pairing isolates the tested λ under this training protocol.

The early criterion passed and the complete final-benefit gate passed. [Status and all five checks](status.md)
