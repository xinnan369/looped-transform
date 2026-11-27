# Frozen E40 midpoint ambiguity and recovery

All four frozen EMA/live-BN runs passed the declared parity checks. Per checkpoint, 40 B internal-validation images contributed 160 stage-image count rows, 16 grouped result rows and 4,295 GT. The two-seed, two-arm tables below describe repeated predictions on the same 40 images. Pooled rows are descriptive totals. [Stage metrics](e40_stage_metrics.csv) · [Seed, arm and density bins](e40_uncertainty_bins.csv) · [Execution and threshold AP](e40_execution_evidence.json)

## Midpoint misses and later recovery

| Midpoint pooled ambiguity | GT observations | Midpoint FN | FN rate | FN recovered at t4 | Recovery among FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0, .25) | 535 | 466 | 87.1% | 21 | 4.5% |
| [.25, .75) | 5,658 | 3,950 | 69.8% | 261 | 6.6% |
| [.75, 1] | 10,987 | 2,660 | 24.2% | 566 | 21.3% |

The highest ambiguity bin has the lowest midpoint FN rate in all four cases (23.4–26.0%). Among its midpoint misses, 18.2–24.8% recover by t4, compared with 3.2–5.7% in the lowest bin. In the highest-density group, the corresponding pooled recovery rates are 274/1,536 (17.8%) and 14/189 (7.4%). The lowest two ambiguity bins contain 4,416 of 7,076 midway misses (62.4%) and account for 282 of 848 recoveries (33.3%). A policy that emphasizes high ambiguity could concentrate on more recoverable misses while giving less weight to most observed misses. The frozen diagnostic cannot identify the causal effect of that policy.

The candidate denominator comprises owned midpoint predictions with score >.30. All 15,687 were assigned to the highest spatial ambiguity bin, including 5,583 FP (35.6%). This bin contained 4,775 duplicates near matched GT and 808 background FP. The lower bins have empty candidate denominators, so these bins cannot distinguish FP propensity. No midpoint FP reached the predeclared high-score cutoff of .75. Neither that zero nor the pooled cell score establishes confidence calibration. Per-image score margins, entropies, match IDs and displacements are retained locally and indexed by checksum. [Local evidence index](epoch_index.csv)

## Same-GT transitions and count tradeoffs

| Seed / arm | t2→t4 recovered / lost GT | t2→t4 Δ MAE | t3→t4 recovered / lost GT | t3→t4 Δ MAE |
| --- | ---: | ---: | ---: | ---: |
| 42 / S4-shift | 180 / 117 | −0.225 | 82 / 79 | +1.000 |
| 42 / U4-shift | 235 / 82 | −0.500 | 260 / 49 | −4.500 |
| 123 / U4-shift | 210 / 122 | −0.700 | 4 / 485 | −23.475 |
| 123 / S4-shift | 223 / 123 | −1.500 | 89 / 69 | +0.150 |

Pooled nAP increased from t2 to t4 and from t3 to t4 in each checkpoint. The 123/U4-shift late read lost 485 GT and recovered 4 while image MAE fell 23.475, illustrating that fixed-threshold matching changes and net count error can move in different directions. Matched-both tracks use GT annotation rows, and candidate-slot changes, displacement, score and rank are reported separately by seed and density. Window-seam association remains unavailable in this frozen diagnostic. [Paired transitions](paired_contrasts.csv)

## Spatial placebo on the frozen maps

| Seed / arm | Valid cells changed | Median per-image alignment correlation | Median per-image gain SD | Gain extrema |
| --- | ---: | ---: | ---: | ---: |
| 42 / S4-shift | 970,150 / 983,040 | −0.0583 | 0.0292 | 0.9581–1.2378 |
| 42 / U4-shift | 970,474 / 983,040 | −0.0716 | 0.0298 | 0.9560–1.2405 |
| 123 / U4-shift | 970,566 / 983,040 | −0.0685 | 0.0302 | 0.9533–1.2393 |
| 123 / S4-shift | 970,380 / 983,040 | −0.0607 | 0.0300 | 0.9547–1.2372 |

The fixed permutation preserved each image's gain multiset and changed about 98.7% of valid cells. Correlation was defined on every image. These hypothetical gains describe the frozen S4-shift and U4-shift control outputs. No guided weights were trained and no guided forward pass was measured here.

The diagnostic warrants review of whether the recovery association justifies a controlled intervention given the concentration of misses at lower ambiguity, the empty candidate bins and heterogeneous t3→t4 outcomes. Spatial specificity, accuracy benefit and sharing benefit remain untested. Shared-2 aux+EMA is the formal baseline.
