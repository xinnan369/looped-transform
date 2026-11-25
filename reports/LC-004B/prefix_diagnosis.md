# LC-004B prefix and count-error diagnosis

All rows use E40. Shared-2 call counts zero and one apply the full-depth head to intermediate features. The separate CNN, Shared-1 and Unshared-2 controls were trained independently. Each row's fitted cutoff comes from the same original train-only fit IDs. All MAE, counts, matches and AP below use internal validation.

| Seed | Model | Calls | Fit cutoff | MAE raw → fitted | Predicted heads raw → fitted | Local FP raw → fitted | Local FN raw → fitted | Fixed-pool local nAP |
|---:|---|---:|---:|---|---|---|---|---:|
| 42 | cnn_only | 0 | 0.31 | 206.30 → 204.37 | 15,782 → 13,250 | 7,899 → 6,218 | 6,294 → 7,145 | 0.18053 |
| 42 | shared_1 | 1 | 0.31 | 224.67 → 228.23 | 16,483 → 13,946 | 8,319 → 6,580 | 6,013 → 6,811 | 0.17144 |
| 42 | shared_2 | 0 | 0.29 | 191.27 → 181.07 | 10,285 → 12,805 | 4,878 → 6,524 | 8,770 → 7,896 | 0.15334 |
| 42 | shared_2 | 1 | 0.33 | 372.73 → 272.93 | 21,155 → 12,531 | 12,730 → 6,466 | 5,752 → 8,112 | 0.13828 |
| 42 | shared_2 | 2 | 0.30 | 218.03 → 218.03 | 15,254 → 15,254 | 7,481 → 7,481 | 6,404 → 6,404 | 0.17268 |
| 42 | unshared_2 | 2 | 0.31 | 222.57 → 213.80 | 15,896 → 13,301 | 7,779 → 6,066 | 6,060 → 6,942 | 0.17505 |
| 123 | cnn_only | 0 | 0.30 | 220.10 → 220.10 | 16,398 → 16,398 | 8,435 → 8,435 | 6,214 → 6,214 | 0.17365 |
| 123 | shared_1 | 1 | 0.31 | 228.87 → 199.60 | 17,691 → 14,853 | 9,050 → 7,080 | 5,536 → 6,404 | 0.18493 |
| 123 | shared_2 | 0 | 0.29 | 213.53 → 193.07 | 9,115 → 11,543 | 4,084 → 5,649 | 9,146 → 8,283 | 0.14993 |
| 123 | shared_2 | 1 | 0.34 | 445.43 → 311.60 | 23,510 → 12,569 | 14,952 → 6,811 | 5,619 → 8,419 | 0.12834 |
| 123 | shared_2 | 2 | 0.31 | 252.20 → 219.67 | 17,817 → 15,099 | 9,241 → 7,331 | 5,601 → 6,409 | 0.17787 |
| 123 | unshared_2 | 2 | 0.31 | 254.00 → 239.00 | 18,605 → 15,699 | 9,865 → 7,799 | 5,437 → 6,277 | 0.16690 |
| 2026 | cnn_only | 0 | 0.31 | 256.60 → 228.27 | 17,363 → 14,907 | 9,276 → 7,506 | 6,090 → 6,776 | 0.17332 |
| 2026 | shared_1 | 1 | 0.31 | 265.37 → 228.83 | 17,786 → 15,240 | 9,366 → 7,541 | 5,757 → 6,478 | 0.17457 |
| 2026 | shared_2 | 0 | 0.29 | 205.30 → 175.00 | 8,846 → 10,807 | 3,783 → 5,013 | 9,114 → 8,383 | 0.15824 |
| 2026 | shared_2 | 1 | 0.33 | 410.20 → 253.77 | 23,711 → 15,546 | 14,719 → 8,409 | 5,185 → 7,040 | 0.15302 |
| 2026 | shared_2 | 2 | 0.31 | 285.43 → 246.10 | 17,892 → 15,246 | 9,598 → 7,669 | 5,883 → 6,600 | 0.17040 |
| 2026 | unshared_2 | 2 | 0.30 | 235.37 → 235.37 | 15,994 → 15,994 | 7,951 → 7,951 | 6,134 → 6,134 | 0.17211 |

There are 14,177 validation GT heads. Count-side TP, FP and FN use greedy one-to-one matches among owned candidates at local δ=0.50. Precision and recall are in the comparison matrix by density and in the per-image CSV. Local nAP uses all in-frame candidates above the fixed 0.05 original floor, including candidates outside midpoint count ownership. The monotonic remapping check preserves the same candidate pool and holds stable ties in source order.

## Gap between independently trained models

| Comparator | Seed | Shared-2 minus comparator raw MAE | Fitted MAE gap | Closure | FP gap raw → fitted | FN gap raw → fitted |
|---|---:|---:|---:|---:|---|---|
| shared_1 | 42 | -6.63 | -10.20 | N/A | -838 → +901 | +391 → -407 |
| shared_1 | 123 | +23.33 | +20.07 | 14.00% | +191 → +251 | +65 → +5 |
| shared_1 | 2026 | +20.07 | +17.27 | 13.95% | +232 → +128 | +126 → +122 |
| unshared_2 | 42 | -4.53 | +4.23 | N/A | -298 → +1,415 | +344 → -538 |
| unshared_2 | 123 | -1.80 | -19.33 | N/A | -624 → -468 | +164 → +132 |
| unshared_2 | 2026 | +50.07 | +10.73 | 78.56% | +1,647 → -282 | -251 → +466 |

## Paired prefix change within Shared-2

| Seed | Count policy | MAE change calls 2−1 | Images improved / degraded / tied | MAE change calls 2−0 | FP change calls 2−1 | FN change calls 2−1 |
|---:|---|---:|---|---:|---:|---:|
| 42 | raw | -154.70 | 20 / 10 / 0 | +26.77 | -5,249 | +652 |
| 42 | calibrated | -54.90 | 18 / 12 / 0 | +36.97 | +1,015 | -1,708 |
| 123 | raw | -193.23 | 23 / 7 / 0 | +38.67 | -5,711 | -18 |
| 123 | calibrated | -91.93 | 19 / 10 / 1 | +26.60 | +520 | -2,010 |
| 2026 | raw | -124.77 | 21 / 8 / 1 | +80.13 | -5,121 | +698 |
| 2026 | calibrated | -7.67 | 19 / 10 / 1 | +71.10 | -740 | -440 |

Across the three E40 seeds, the mean two-minus-one MAE difference was -157.57 at the archived cutoff and -51.50 after separate train-fit cutoffs. LC-004A found lower two-call MAE in six best/last snapshots at 0.30 and one-call overcounts in all six. These prefix responses reflect the distribution presented to a full-depth trained head.

## Density groups and candidate floors

| Shared-2 E40 seed | Density | Images | Two-call MAE raw → fitted | Local precision raw → fitted | Local recall raw → fitted |
|---:|---|---:|---|---|---|
| 42 | low | 11 | 95.91 → 95.91 | 48.00% → 48.00% | 52.35% → 52.35% |
| 42 | mid | 8 | 258.12 → 258.12 | 48.95% → 48.95% | 48.29% → 48.29% |
| 42 | high | 11 | 311.00 → 311.00 | 52.39% → 52.39% | 58.17% → 58.17% |
| 123 | low | 11 | 118.73 → 100.27 | 42.67% → 46.81% | 56.26% → 51.05% |
| 123 | mid | 8 | 317.12 → 302.12 | 45.74% → 48.91% | 50.88% → 45.16% |
| 123 | high | 11 | 338.45 → 279.09 | 50.29% → 53.39% | 65.54% → 59.73% |
| 2026 | low | 11 | 136.82 → 123.36 | 39.48% → 42.04% | 52.56% → 48.05% |
| 2026 | mid | 8 | 289.12 → 276.62 | 45.08% → 48.49% | 47.98% → 42.92% |
| 2026 | high | 11 | 431.36 → 346.64 | 48.39% → 51.88% | 64.33% → 59.15% |

| Shared-2 E40 seed | Original >0.05 local nAP | Raw >0.10 diagnostic | Raw >0.20 diagnostic |
|---:|---:|---:|---:|
| 42 | 0.17268 | 0.17212 | 0.16678 |
| 123 | 0.17787 | 0.17724 | 0.17218 |
| 2026 | 0.17040 | 0.16985 | 0.16409 |

Raising the diagnostic floor changes how many predictions can receive a GT match. The original ranks among retained candidates stay fixed. The fitted count cutoff retains the original fixed-pool nAP. Its measured effect is on counting.

These observations identify a mismatch at the intermediate head. They motivate testing whether an intermediate loss would improve that prefix, but provide no direct estimate of the effect of such training. A separate study would need the same two-call core and head, a fixed first-call loss weight versus final-only loss, and a matched optimizer, update budget and data split.

[Per-image raw and calibrated counts](per_image_counts.csv) · [Paired image changes](paired_image_changes.csv) · [All raw-floor sensitivity results](score_floor_diagnostics.csv)
