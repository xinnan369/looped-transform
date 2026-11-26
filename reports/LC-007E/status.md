# LC-007E four-call depth screen

**CPU_PREPARATION_COMPLETE / GPU_TRAINING_NOT_AUTHORIZED.** The implementation and preregistration froze at `2dc472c602eeb09c646e1085547a8d716e1136ab` on the reviewed LC-007D result `ae903e2467648059e1d466bde011c2396e44eada`. Issue #23 now links the original dated LC-007D authorization and the completed engineering review. The earlier negative research gate triggered the separate K=4 proposal. [Issue #25](https://github.com/xinnan369/looped-transform/issues/25) · [Reconciled Issue #23](https://github.com/xinnan369/looped-transform/issues/23) · [LC-007D PR #24](https://github.com/xinnan369/looped-transform/pull/24)

| Deliverable | Evidence |
| --- | --- |
| Shared/unshared × ungated/RS factorial | [Architecture and FLOPs](architecture.md) |
| Legacy API compatibility, unit-gate equivalence, gradients, EMA and interruption | [CPU tests](test_results.md) |
| Eight initial states, cached inputs and historical pair checks | [Initialization audit](initialization_audit.json) |
| Fixed recipe, depth/sharing gates and readouts | [Protocol](protocol.md) · [Stagewise plan](stagewise_analysis.md) |
| Proposed budget, current execution and next decision | [Resource plan](resource_plan.md) · [Resource audit](resource_audit.md) · [Next steps](next_steps.md) |

All eight project runs and experimental outcomes are `not_run`. The initial proposal is capped at 9,000 charged seconds and 2 GiB process peak reserved, with zero new downloads. The proposal needs separate owner approval. [Epoch index](epoch_index.csv) · [Paired contrasts](paired_contrasts.csv) · [Gate statistics](gate_statistics.csv) · [Research decision](research_decision.json)
