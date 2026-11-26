# LC-007D channel gate screen

**CPU_PREPARATION_COMPLETE / GPU_TRAINING_NOT_AUTHORIZED.** The draft review covers the fixed-K channel gates, local synthetic verification, cached-input and historical-reference checks and a preregistered B screen. Code and protocol froze at `6f6c37a2c3643e1b97ec0d1f27d53cdf2b60b695`, descending from the LC-007C authorization reconciliation at `5751f1005ce3ff22db554374e0e9fd55b58a646a`.

| Deliverable | Evidence |
| --- | --- |
| R, RS and matched U-RS architecture | [Equations, parameter counts and FLOPs](architecture.md) |
| Identity initialization, FP32/BF16 loss/output/gradient parity and legacy regression | [Local test results](test_results.md) |
| Local cache hashes, historical trajectory and metric audit, CPU profiling | [Initialization audit](initialization_audit.json) |
| Frozen execution plan and planned readouts | [Protocol](protocol.md) · [Resource plan](resource_plan.md) · [Stagewise analysis](stagewise_analysis.md) |
| Current resource use and research decision | [Resource audit](resource_audit.md) · [Decision](research_decision.json) · [Next decision](next_steps.md) |

The [run index](epoch_index.csv), [contrast index](paired_contrasts.csv) and [gate statistics index](gate_statistics.csv) reflect `not_run` for experimental work. Six real runs, accuracy claims and the GPU latency and memory audit depend on the separately approved GPU scope. The B comparisons use eight historical fixed controls whose actual reference reuse is pending the new runs' checks.

[LC-007D Issue #23](https://github.com/xinnan369/looped-transform/issues/23) · [LC-007C authorization record](../LC-007C/authorization_record.md) · [LC-007C draft PR #22](https://github.com/xinnan369/looped-transform/pull/22)
