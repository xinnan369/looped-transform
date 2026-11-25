# LC-004B status

The frozen diagnostic evaluated all 18 E40 checkpoint/call combinations from 12 LC-003 model snapshots. All cutoffs were selected on the same 60 original training images and frozen before the single internal-validation pass per case. No model parameters were updated. Measured fit and validation work took 181.4 seconds on one RTX 3090, excluding preflight.

**Outcome. The preregistered two-comparator gate did not support the score-calibration explanation.**

| Comparison | Mean raw MAE gap | Mean fitted MAE gap | Closure | Positive raw gaps narrowed | Gate |
|---|---:|---:|---:|---:|:---:|
| Shared-2 minus shared_1 | +12.26 | +9.04 | 26.20% | 2/3 | fail |
| Shared-2 minus unshared_2 | +14.58 | -1.46 | 109.98% | 1/3 | fail |

Signed MAE gaps use the same validation images. Positive gaps favor the comparator. Seed-level closure is N/A when raw Shared-2 MAE was already equal or lower. The mean uses all three E40 seeds with equal weight. The gate requires at least 50% mean closure, two narrowed positive seed gaps and invariant fixed-pool rank AP.

The Shared-1 comparison narrowed both positive raw seed gaps and closed 26.20% of the mean raw gap. For Unshared-2, seeds 42 and 123 already favored Shared-2 in the raw comparison. The large improvement at seed 2026 reversed the mean gap and supplied the only narrowed positive raw gap.

Fitted cutoffs ranged from 0.29 to 0.34. There were 0 tied fit objectives at the minimum. The threshold table retains each complete grid and all minimizing points.

Fixed-pool rank and local nAP invariance across all 18 cases was **verified**. All raw E40 counts and local nAP replayed LC-004A, and the per-image candidate hashes matched before and after evaluating the count masks. The separate score-floor diagnostic measures a change in candidate membership.

| Deliverable | Evidence |
|---|---|
| Protocol and fit selection | [Frozen protocol](calibration_protocol.md) · [60 selected IDs](fit_manifest.json) |
| Fitted values and all ties | [Thresholds, exact objective sums and timing](calibration_thresholds.csv) |
| Fixed-pool metrics | [Comparison matrix](comparison_matrix.csv) · [Per-image counts](per_image_counts.csv) |
| Paired comparisons | [Prefix diagnosis](prefix_diagnosis.md) · [MAE gaps](paired_mae_gaps.csv) · [Image pairs](paired_image_changes.csv) |
| Rank invariance and floor diagnostic | [Decision](calibration_decision.json) · [Raw floor sensitivity](score_floor_diagnostics.csv) |
| Provenance and resources | [Hashes and run metadata](calibration_evidence.json) · [Tests](test_results.md) · [Download audit](download_audit.md) |

The preregistered protocol commit is `a7b5d14be80aad4ac73723661328071cb7570ca1`. The fit uses images already seen by the model during training. All results are exploratory and use the original internal validation set.

[Issue #11](https://github.com/xinnan369/looped-transform/issues/11) · [PR #10](https://github.com/xinnan369/looped-transform/pull/10) · [Follow-up](next_steps.md)
