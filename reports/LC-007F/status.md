# LC-007F eight-call depth screen

**GPU_SCREEN_COMPLETE / READY_FOR_REVIEW.** All eight authorized K=8 runs completed on 2026-11-26 with 320 epochs and 57,600 successful updates. The executed head was `d041b3473ed270234c2277ea22ad0b410d7d3ff7`, containing the implementation freeze `60c6c2a5f3f24a9fc8c364b11ae7154e1e231777`. [Owner approval](authorization.json) · [Run evidence](run_evidence.json) · [Issue #27](https://github.com/xinnan369/looped-transform/issues/27) · [Issue #25](https://github.com/xinnan369/looped-transform/issues/25) · [PR #26](https://github.com/xinnan369/looped-transform/pull/26)

| K=8 final EMA E36–40 | MAE | Local nAP |
| --- | ---: | ---: |
| S8 | 24.2000 | 0.297760 |
| U8 | 23.3800 | 0.301183 |
| S8RS | 25.0200 | 0.298882 |
| U8RS | 23.8900 | 0.302734 |

All eight joint gates failed. S8 reduced MAE by 0.1875 relative to S4 and lost 0.002675 local nAP. S8RS increased MAE by 0.9700 relative to S4RS and gained 0.000401 local nAP. Both shared K=8 arms had worse MAE than the corresponding K=2 controls and missed the macro RMSE guard on those comparisons. Both also had higher MAE and lower local nAP than their same-depth unshared controls. Retain ungated Shared-2 + auxiliary supervision + EMA. [Preregistered decisions](research_decision.json) · [Paired results](paired_contrasts.csv)

Relative to K=4, ungated and RS sharing gaps widened on both metrics. Relative to K=2, the ungated MAE gap narrowed by 0.2600 and the local nAP gap narrowed by 0.001726. The RS MAE gap widened by 0.2225 and its nAP gap narrowed by 0.001297. These are descriptive interactions across depths with different compute and auxiliary positions.

| Deliverable | Evidence |
| --- | --- |
| K=8 shared/unshared and ungated/RS factorial | [Architecture and costs](architecture.md) |
| FP32/BF16 parity, scheduler invariance and legacy regression | [Local tests](test_results.md) |
| Historical pairing, cached data and eight new starts | [Initialization and resource audit](initialization_audit.json) |
| Two-read and late eight-read policy, joint gates | [Protocol](protocol.md) · [Stagewise metrics](stagewise_analysis.md) |
| Proposed budget and current activity | [Resource plan](resource_plan.md) · [Executed resources](resource_audit.md) |

All 24 indexed result files and 77,760 rows reconciled. GPU time charged 7,646.81 seconds and peak reserved was 1,260,388,352 bytes. The results cover two seeds on internal ShanghaiTech B validation. [Epoch index](epoch_index.csv) · [Gate statistics](gate_statistics.csv) · [Resource audit](resource_audit.md) · [Next steps](next_steps.md)
