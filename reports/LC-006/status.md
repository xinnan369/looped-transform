# LC-006 status

Completed 18 fresh 40-epoch runs on one RTX 3090 with 129,600 successful updates, zero AMP skips and 102.2 minutes of measured epoch training plus validation. Seed-specific actual crop orders and learning-rate arrays match across all six arms. Data and settings froze before training.

**The preregistered B replication and repeated-depth gates both failed.** Shared-2 auxiliary supervision reduced mean final MAE by 3.56% in three of three seeds. Four of the six shared runs exceeded the stability CV limit. The supervised Shared-2 final MAE was 0.34 heads higher than independently trained Shared-1. Counts and local nAP come from internal validation.

| Task | Evidence |
|---|---|
| T1 | Original Part B train360 and internal-val40, raw and cleaned point replay, split IDs, decoded images and stride4/K2 capacity. [Data audit](data_audit.json) · [Protocol](protocol.md) |
| T2 | Shared and independent two-call graphs, matched A/B full hashes, cross-architecture common and first-block hashes and matched timing inputs. [Preflight and profiles](preflight.json) |
| T3 | All 18 arms with best selected by earliest final MAE. [Matrix](experiment_matrix.csv) · [Per-run epoch index](training_history.csv) · [Execution evidence](run_evidence.json) |
| T4 | Replication, supervision, depth and sharing contrasts. [Paired results](paired_seed_results.csv) · [Decision](research_decision.json) · [Analysis](depth_supervision_analysis.md) |
| T5 | Three local Python test suites and zero external experiment bytes. [Tests](test_results.md) · [Resource audit](download_audit.md) |

## Verified data and capacity

| Split | Images | Raw heads | Cleaned heads | Rejected | Stride4/K2 full-image overflow |
|---|---:|---:|---:|---:|---:|
| train | 360 | 44974 | 44856 | 118 | 8 |
| val | 40 | 4306 | 4295 | 11 | 1 |

Train density boundaries are 82.65 and 180.99 heads per megapixel. The fixed seed-42 epoch-one 512 px crop probe has zero overflow in both splits. Actual per-epoch sampled and overflow counts are archived with the training crops. Only the original official-train images were opened.

## Internal-validation outcomes

| Arm | Final late MAE mean ± sample SD | Final late RMSE | Final late local nAP | Stability passes |
|---|---:|---:|---:|---:|
| cnn | 28.28 ± 2.80 | 37.71 | 0.27479 | 0/3 |
| shared_1 | 25.42 ± 0.14 | 33.68 | 0.28956 | 0/3 |
| shared_2_final_only | 26.71 ± 0.35 | 35.43 | 0.29230 | 1/3 |
| shared_2_aux_025 | 25.77 ± 1.05 | 34.56 | 0.29254 | 1/3 |
| unshared_2_final_only | 24.98 ± 0.75 | 33.58 | 0.29474 | 1/3 |
| unshared_2_aux_025 | 24.70 ± 0.94 | 33.51 | 0.29647 | 1/3 |

Runs outside the stability gate are 42/cnn (CV 11.10%, final loss -1.78%), 42/shared_1 (CV 11.41%, final loss -2.26%), 123/cnn (CV 12.65%, final loss -1.52%), 123/shared_1 (CV 15.37%, final loss -1.58%), 123/shared_2_final_only (CV 20.13%, final loss -1.65%), 123/shared_2_aux_025 (CV 16.14%, final loss -1.74%), 123/unshared_2_final_only (CV 13.39%, final loss -1.84%), 123/unshared_2_aux_025 (CV 12.49%, final loss -1.83%), 2026/cnn (CV 12.08%, final loss -1.02%), 2026/shared_1 (CV 21.02%, final loss -1.35%), 2026/shared_2_final_only (CV 15.46%, final loss -1.43%), 2026/shared_2_aux_025 (CV 11.60%, final loss -1.39%), 2026/unshared_2_final_only (CV 14.64%, final loss -1.35%), 2026/unshared_2_aux_025 (CV 20.07%, final loss -1.47%).

Training and preflight used code `3929926602df3ef1af1ac22508bba27e728a8117`. The preflight records the workspace, runtime, commands and exact model and data hashes.

[Issue #15](https://github.com/xinnan369/looped-transform/issues/15) · [PR #14](https://github.com/xinnan369/looped-transform/pull/14) · [Follow-up](next_steps.md)
