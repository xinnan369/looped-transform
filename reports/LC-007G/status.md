# LC-007G four-call cross-window propagation

**GPU_SCREEN_COMPLETE / READY_FOR_REVIEW.** All four authorized runs completed on 2026-11-26, with 160 epochs and 28,800 successful updates. The executed head was `3e01e677714f87d71a2f7b0518a48f71f2669f82`, containing implementation freeze `62b336bcf0bdeea9e5fa66ded2cd8c5e81fb4926`. [Authorization](authorization.json) · [Run evidence](run_evidence.json) · [Issue #29](https://github.com/xinnan369/looped-transform/issues/29) · [PR #28](https://github.com/xinnan369/looped-transform/pull/28)

| Final EMA E36–40 | MAE | Local nAP |
| --- | ---: | ---: |
| S4 fixed | 24.3875 | 0.300434 |
| S4-shift | 22.6650 | 0.305224 |
| U4 fixed | 23.9200 | 0.301280 |
| U4-shift | 20.7900 | 0.305681 |
| Retained S2 | 21.9375 | 0.295321 |

S4-shift and U4-shift both passed their joint geometry gates, with both metrics favorable in both seeds. Their MAE fell 7.06% and 13.09% from the matched fixed controls. The direct shifted sharing gate failed. The shared-minus-independent MAE gap widened by 1.4075, and the local nAP gap narrowed by 0.000389. S4-shift's MAE remained 0.7275 higher than the retained S2 and missed the 3% target, although local nAP rose by 0.009904. Retain ungated Shared-2 with auxiliary supervision and EMA. [Decision checks](research_decision.json) · [Paired results](paired_contrasts.csv)

| Deliverable | Evidence |
| --- | --- |
| Existing shifted Shared-4 with opt-in stage reads and matched independent arm | [Architecture and FLOPs](architecture.md) |
| FP32/BF16 token influence and gradient paths | [Cross-window mechanism](cross_window_mechanism.md) |
| Legacy regression, first-stage parity, masking and EMA | [Test results](test_results.md) |
| Cached initialization and paired history | [Initialization audit](initialization_audit.json) |
| Frozen B screen and decision gates | [Protocol](protocol.md) · [Real-data stagewise plan](stagewise_analysis.md) |
| Time budget and observational memory policy | [Resource plan](resource_plan.md) · [Executed resources](resource_audit.md) |

The archive verifies 12 indexed result files and 56,480 rows. The run charged 3,609.74 seconds and reserved at most 983,564,288 bytes, with observational memory accounting and zero downloads. The two-seed B screen and the synthetic mechanism probe support distinct conclusions. Candidate identities remain unavailable for seam-localized recall. [Run index](epoch_index.csv) · [Stagewise results](stagewise_analysis.md) · [Next steps](next_steps.md)
