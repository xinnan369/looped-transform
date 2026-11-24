# LC-003 status

LC-003 completed 17 fixed 40-epoch runs and 4 CUDA profiles using one RTX 3090. Measured training and epoch validation totaled 69.0 minutes, excluding profiles, checkpoint writes and best-checkpoint re-evaluations. The experiment continues from PR #6 head `8bb2de29240991003d746d3dc3175b44b931cada` and uses the frozen ShanghaiTech A training and internal validation splits.

The preregistered rule selected **`warmup_cosine`** for all two-call controls. At seed 42, the fixed-cutoff last-five mean MAEs were 216.27 for one-call Shared-1, 218.77 for Shared-2 and 213.12 for its independent control. The shifted pair recorded 230.29 and 209.14, respectively.

Seeds 123 and 2026 completed for the matched `shared_2` comparison and both baselines. The report treats between-seed variation descriptively.

| Task | Evidence |
|---|---|
| T1 · Diagnose the 20-epoch archive | Seven checkpoint rechecks, grouped validation errors, 140 epoch records and the remaining hypotheses. [Diagnosis](diagnosis.md) |
| T2 · Compare two training schedules | Six seed-42 runs with 5,400 updates each, fixed .30 count threshold and checkpoint-resume equivalence. [Schedule comparison](schedule_ablation.md) |
| T3 · Match two shared and independent calls | Four seed-42 arms under one schedule. The identical Shared-2 run was reused from T2. [Experiment matrix](experiment_matrix.csv) |
| T4 · Decide seed replication | The preregistered gate controls the additional runs. [Paired-seed results](paired_seed_results.csv) |
| T5 · Verify and archive | 66 tests passed on each of Python 3.10, 3.11 and 3.13. [Tests](test_results.md) · [Epochs and provenance](training_history.json) |
| Resources | 0 external experiment bytes, no hosted matrix, NWPU paused. [Download audit](download_audit.md) |

All results refer to internal validation. Local nAP uses the frozen spacing-relative matching protocol. [Issue #7](https://github.com/xinnan369/looped-transform/issues/7) · [PR #6](https://github.com/xinnan369/looped-transform/pull/6) · [Follow-up](next_steps.md)
