# LC-007D channel gate screen

**GPU_SCREEN_COMPLETE / READY_FOR_REVIEW.** All six fresh runs completed on 2026-11-26 in the authorized order. The full 240 epochs and 43,200 successful updates passed archive checks. The runner executed PR #24 head `99851b1baf893a6619bbe79c980c8635c9ac896a`, containing the code and protocol freeze `6f6c37a2c3643e1b97ec0d1f27d53cdf2b60b695`. [Owner approval](authorization.json) · [Run evidence](run_evidence.json)

On final EMA E36–40, R improved MAE by 0.71% and local nAP by 0.002239 compared with ungated Shared-2. RS increased MAE by 1.26% and improved local nAP by 0.000345. Both arms fell short of the preregistered 3% MAE reduction. RS also fell short of the matched U-RS sharing challenge on both metrics. The decision retains Shared-2 + auxiliary loss + EMA. [Frozen decisions](research_decision.json) · [Paired contrasts](paired_contrasts.csv)

| Deliverable | Evidence |
| --- | --- |
| R, RS and matched U-RS architecture | [Equations, parameter counts and FLOPs](architecture.md) |
| Identity initialization, FP32/BF16 loss/output/gradient parity and legacy regression | [Local test results](test_results.md) |
| Local cache hashes, historical trajectory and metric audit, CPU profiling | [Initialization audit](initialization_audit.json) |
| Frozen execution plan and planned readouts | [Protocol](protocol.md) · [Resource plan](resource_plan.md) · [Stagewise analysis](stagewise_analysis.md) |
| Current resource use and research decision | [Resource audit](resource_audit.md) · [Decision](research_decision.json) · [Next decision](next_steps.md) |

The [run index](epoch_index.csv) verifies 18 files and 42,480 rows. The [gate statistics](gate_statistics.csv) cover all stages, branches and readouts. The eight fixed historical controls passed the reuse checks and were trained in the earlier LC-007B screen. This is a two-seed Part B screen.

[LC-007D Issue #23](https://github.com/xinnan369/looped-transform/issues/23) · [LC-007C authorization record](../LC-007C/authorization_record.md) · [LC-007C draft PR #22](https://github.com/xinnan369/looped-transform/pull/22)
