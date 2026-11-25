# LC-007A accuracy diagnosis

The fixed E40 replay and historical audit were completed from PR #16 head `7848efc8dd572edb5de43b2f3aaa0e5eff46f209`. All 33 checkpoint passes ran once on a local RTX 3090 (15 A, 18 B), covering 1,170 checkpoint/image visits. Staged inference produced 51 readouts, all aligned with their archival metrics. No frozen case was missing. The measured sum of per-checkpoint execution was 140.9 s and the largest peak tensor allocation was 187,772,928 bytes.

| Task | Outcome |
| --- | --- |
| Historical reconciliation | 47 prior fresh trajectories. Recomputed 1,640 epochs across LC-003/005/006 with unchanged archive decisions. B failed the CV gate for 14/18, including all CNN runs. [Ledger](run_ledger.csv) · [Stability diagnosis](stability_diagnosis.md) |
| Paired error attribution | Shared aux helped count error relative to shared λ0 on both A and B. At E40 B it had 3.94 and 3.02 higher MAE than independent Shared-1 and same-supervision Unshared-2 respectively, with the largest deficits at high density. A lacks the equally supervised unshared control. [Paired diagnosis](error_attribution.md) |
| Image sampling | The two B independent-comparator deficits retained their direction for every one-image omission. A's E40 Shared-1 difference changed sign for some omissions. Compare image sensitivity with the separate training-seed deltas and sample SD. [Sampling evidence](sampling_sensitivity.json) |
| Future proposal | One fixed parameter EMA screen for all representative arms, with live-vs-shadow comparisons, a stopping rule and a staged budget. LC-007B remains a draft for owner review. [Hypotheses](hypothesis_register.md) · [Proposed protocol](lc007b_proposed_protocol.md) |

The historical results and E40 diagnostics do not support a stable recurrent accuracy advantage over the fair controls. The inference code did not estimate EMA effects. The observed MAE, coverage and local nAP are internal development-validation endpoints.

The [protocol](protocol.md) specifies commands, cases and fixed scoring. The [frozen case index](frozen_cases.csv) and [replay provenance](replay_execution.json) pin the local checkpoint inputs and execution commit `747828df4e8b9b4a5c5db4b9aa19add005126bdf`. [Per-run image and contrast file index](paired_image_metrics.csv) lists paths, exact byte sizes, SHA-256 values and status. See the [test results](test_results.md), [resource audit](download_audit.md) and [next steps](next_steps.md). [Issue #17](https://github.com/xinnan369/looped-transform/issues/17)
