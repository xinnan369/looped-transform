# LC-007G four-call cross-window propagation

**CPU_PREPARATION_COMPLETE / GPU_TRAINING_NOT_AUTHORIZED.** The implementation and preregistration froze at `62b336bcf0bdeea9e5fa66ded2cd8c5e81fb4926` on reviewed K=8 head `8b2e912c2591cac643cb3d93732c870f5674fd50`. Issue #27 already records the dated approval and completed negative K=8 result. The new proposal returns to Shared-4 and compares fixed and shifted geometry within both sharing regimes. [Issue #29](https://github.com/xinnan369/looped-transform/issues/29) · [Issue #27](https://github.com/xinnan369/looped-transform/issues/27) · [PR #28](https://github.com/xinnan369/looped-transform/pull/28)

| Deliverable | Evidence |
| --- | --- |
| Existing shifted Shared-4 with opt-in stage reads and matched independent arm | [Architecture and FLOPs](architecture.md) |
| FP32/BF16 token influence and gradient paths | [Cross-window mechanism](cross_window_mechanism.md) |
| Legacy regression, first-stage parity, masking and EMA | [Test results](test_results.md) |
| Cached initialization and paired history | [Initialization audit](initialization_audit.json) |
| Frozen B screen and decision gates | [Protocol](protocol.md) · [Real-data stagewise plan](stagewise_analysis.md) |
| Time budget and observational memory policy | [Resource plan](resource_plan.md) · [Executed resources](resource_audit.md) |

The synthetic probe observes cross-window influence and gradients at call two and farther propagation at calls three and four. The planned B outcomes are `not_run`. Four fresh runs and 28,800 updates are proposed within 90 minutes with zero external downloads. Device memory is observed without a fixed process quota. [Run index](epoch_index.csv) · [Contrasts](paired_contrasts.csv) · [Research decision](research_decision.json) · [Next steps](next_steps.md)
