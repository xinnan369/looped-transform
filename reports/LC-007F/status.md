# LC-007F eight-call depth screen

**CPU_PREPARATION_COMPLETE / GPU_TRAINING_NOT_AUTHORIZED.** The implementation and preregistration froze at `60c6c2a5f3f24a9fc8c364b11ae7154e1e231777` on reviewed K=4 result `143ee08788a337566cfe182ede1d24ba912faaae`. Issue #25 records the separate prior approval and the negative K=4 engineering review. [Issue #27](https://github.com/xinnan369/looped-transform/issues/27) · [Issue #25](https://github.com/xinnan369/looped-transform/issues/25) · [PR #26](https://github.com/xinnan369/looped-transform/pull/26)

| Deliverable | Evidence |
| --- | --- |
| K=8 shared/unshared and ungated/RS factorial | [Architecture and costs](architecture.md) |
| FP32/BF16 parity, scheduler invariance and legacy regression | [Local tests](test_results.md) |
| Historical pairing, cached data and eight new starts | [Initialization and resource audit](initialization_audit.json) |
| Two-read and late eight-read policy, joint gates | [Protocol](protocol.md) · [Stagewise metrics](stagewise_analysis.md) |
| Proposed budget and current activity | [Resource plan](resource_plan.md) · [Executed resources](resource_audit.md) |

All eight project runs and experimental outcomes are `not_run`. The proposed queue is 57,600 successful updates, at most 10,800 charged seconds and 2 GiB process peak reserved, with zero external experiment downloads. [Epoch index](epoch_index.csv) · [Contrast index](paired_contrasts.csv) · [Gate statistics](gate_statistics.csv) · [Research decision](research_decision.json) · [Next steps](next_steps.md)
