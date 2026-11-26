# LC-007E four-call depth screen

**GPU_SCREEN_COMPLETE / READY_FOR_REVIEW.** All eight authorized runs completed on 2026-11-26 with 320 epochs and 57,600 successful updates. Execution used reviewed head `b6d3297a3c37afe302e29551bbda1170117c4794`, containing implementation freeze `2dc472c602eeb09c646e1085547a8d716e1136ab`. [Approval](authorization.json) · [Run evidence](run_evidence.json) · [Issue #25](https://github.com/xinnan369/looped-transform/issues/25) · [Reconciled Issue #23](https://github.com/xinnan369/looped-transform/issues/23) · [LC-007D PR #24](https://github.com/xinnan369/looped-transform/pull/24)

| Final EMA E36–40 comparison | MAE, candidate / reference | Local nAP, candidate / reference | Joint gate |
| --- | ---: | ---: | --- |
| S4 vs Shared-2 | 24.3875 / 21.9375 | 0.300434 / 0.295321 | Fail |
| S4RS vs Shared-2-RS | 24.0500 / 22.2150 | 0.298481 / 0.295666 | Fail |
| S4 vs U4 | 24.3875 / 23.9200 | 0.300434 / 0.301280 | Fail |
| S4RS vs U4RS | 24.0500 / 22.9975 | 0.298481 / 0.300984 | Fail |

Both shared K=4 models raised local nAP relative to their matched K=2 references and increased count MAE. Both depth comparisons also missed the macro RMSE guard. Within K=4, both shared arms had higher MAE and lower local nAP than their matched unshared arms. Stability passed on all four joint comparisons. Retain ungated Shared-2 + auxiliary supervision + EMA.

The ungated sharing gap narrowed by 0.6125 MAE and 0.004304 local nAP. In the RS stratum, the gap widened by 0.1450 MAE and narrowed by 0.002647 local nAP. These are depth-by-sharing interactions on the same data and differing compute. [Paired contrasts](paired_contrasts.csv) · [Frozen decision](research_decision.json)

| Deliverable | Evidence |
| --- | --- |
| Shared/unshared × ungated/RS factorial | [Architecture and FLOPs](architecture.md) |
| Legacy API compatibility, unit-gate equivalence, gradients, EMA and interruption | [CPU tests](test_results.md) |
| Eight initial states, cached inputs and historical pair checks | [Initialization audit](initialization_audit.json) |
| Fixed recipe, depth/sharing gates and readouts | [Protocol](protocol.md) · [Stagewise plan](stagewise_analysis.md) |
| Proposed budget, current execution and next decision | [Resource plan](resource_plan.md) · [Resource audit](resource_audit.md) · [Next steps](next_steps.md) |

The archive verified 24 indexed files with 112,960 rows. Runtime charged 7,385.73 seconds and peak process reserved was 970,981,376 bytes. The two-seed findings apply to exploratory ShanghaiTech B internal validation. [Epoch index](epoch_index.csv) · [Gate statistics](gate_statistics.csv) · [Resource audit](resource_audit.md)
