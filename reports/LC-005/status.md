# LC-005 status

Completed six 40-epoch runs on one RTX 3090. The experiments completed 32,400 actual optimizer updates with zero AMP skips and recorded 36.5 minutes of epoch training and validation.

**The prespecified gate supports a preliminary final-output counting benefit on internal validation.** All six runs use the same Shared-2 model and point head. Training decodes both stages. The only within-seed configuration difference is λ=0.0 or λ=0.25 in the early loss.

| Task | Evidence |
|---|---|
| T1 | Shared block and head, equal parameter counts, zero-weight equivalence, checkpoint and resume tests. [Tests](test_results.md) |
| T2 | All preflight pairs match for initial state, autograd graph, crop probe, decoder calls and LR schedule. [Preflight](preflight.json) · [Frozen protocol](protocol.md) |
| T3 | Six serial runs, 240 epochs, matched per-epoch crop-order hashes and actual LR arrays. [Experiment matrix](experiment_matrix.csv) · [Per-epoch history](training_history.csv) · [Hashes and commands](run_evidence.json) |
| T4 | E36–40 paired analysis and full five-condition gate. [Paired seed results](paired_seed_results.csv) · [Decision](research_decision.json) |
| T5 | Local Python 3.10/3.11/3.13 tests and zero external experiment resource bytes. [Verification](test_results.md) · [Resource audit](download_audit.md) |

## Primary endpoint and stability

MAE is measured at strict score >0.30 on internal validation. Checkpoint selection uses final output only. Local nAP retains a 0.05 candidate floor and the fixed local matching protocol.

| Seed | Arm | Early last-five MAE | Final last-five MAE | Final last-five RMSE | Final local nAP | Final MAE CV | Final train loss change | Stable |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 42 | final_only | 398.39 | 218.77 | 342.18 | 0.17648 | 9.95% | -0.82% | pass |
| 42 | aux_025 | 222.77 | 222.42 | 351.79 | 0.17842 | 8.99% | -0.81% | pass |
| 123 | final_only | 417.11 | 230.53 | 360.66 | 0.18536 | 5.68% | -0.76% | pass |
| 123 | aux_025 | 215.11 | 213.94 | 339.59 | 0.18344 | 3.70% | -0.77% | pass |
| 2026 | final_only | 350.97 | 253.10 | 390.02 | 0.18140 | 8.22% | -0.60% | pass |
| 2026 | aux_025 | 219.39 | 227.33 | 353.87 | 0.17611 | 9.12% | -0.63% | pass |

| Gate | Observation | Result |
|---|---|:---:|
| Final MAE ≥3% better | +5.51% | pass |
| Final MAE better in ≥2 seeds | 2/3 | pass |
| Final local nAP change ≥−0.005 | -0.00176 | pass |
| All six stable | all six pass | pass |
| Early MAE improvement | +43.65% and 3/3 seeds | pass |

## Three-seed summary

| Endpoint | Final-only mean ± sample SD | Aux mean ± sample SD | Paired difference mean ± sample SD |
|---|---:|---:|---:|
| Final last-five MAE | 234.14 ± 17.44 | 221.23 ± 6.78 | -12.90 ± 15.05 |
| Early last-five MAE | 388.82 ± 34.09 | 219.09 ± 3.84 | -169.73 ± 35.58 |
| Final last-five RMSE | 364.29 ± 24.12 | 348.41 ± 7.71 | -15.87 ± 23.32 |
| Final last-five local nAP | 0.18108 ± 0.00445 | 0.17932 ± 0.00375 | -0.00176 ± 0.00362 |
| Early last-five local nAP | 0.14306 ± 0.01404 | 0.17734 ± 0.00352 | +0.03428 ± 0.01384 |

Paired changes are aux minus final-only. The three seeds support an exploratory comparison. See [early and final diagnostics](early_final_diagnostics.md) for paired FP/FN, timing, memory and the historical baseline context.

Training code and successful preflight used `a0980795e425d8ccbaae0d400ace31efa06b1687`. All model and config digests, observed commands and checkpoint hashes are available in the local-run index.

[Issue #13](https://github.com/xinnan369/looped-transform/issues/13) · [PR #12](https://github.com/xinnan369/looped-transform/pull/12) · [Follow-up](next_steps.md)
