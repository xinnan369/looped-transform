# LC-007H training results

`COMPLETE / READY_FOR_REVIEW`. All six frozen runs completed in one attempt, with 240 epochs and 43,200 successful optimizer and EMA updates. The CPU archiver verified checkpoint, initialization, crop/LR, image coverage and metric provenance against the frozen inputs and historical references. [Authorization](authorization.json) · [Runtime preflight](preflight_cpu.json) · [Run evidence](run_evidence.json)

## Preregistered comparisons

The primary endpoint is final EMA E36–40, weighting the five epochs equally within each seed and both seeds equally in the macro. Historical shifted references were measured in earlier runs.

| Candidate | Reference | MAE, candidate / reference | Local nAP, candidate / reference | Joint gate |
| --- | --- | ---: | ---: | --- |
| S-guide | S4-shift | 23.0375 / 22.6650 | .304948660 / .305224300 | Fail |
| U-guide | U4-shift | 21.5375 / 20.7900 | .305516788 / .305681402 | Fail |
| S-guide | S-placebo | 23.0375 / 21.9800 | .304948660 / .304137396 | Fail |
| S-guide | U-guide | 23.0375 / 21.5375 | .304948660 / .305516788 | Fail |

S-guide improved MAE and nAP against S4-shift for seed 42. For seed 123, MAE went from 21.4550 to 23.0850 and nAP went from .30491154 to .30260395. S-guide had a small macro nAP gain against S-placebo and higher macro MAE. Both the spatial specificity and sharing comparisons missed their joint gates. Stability and the FP/FN guards passed in all four comparisons. The RMSE guard failed for U-guide vs U4-shift, S-guide vs S-placebo and S-guide vs U-guide. The S2 parity and 3% counting targets also failed. Shared-2 aux+EMA remains the formal baseline. [Decision checks](research_decision.json) · [Seed, endpoint and density contrasts](paired_contrasts.csv)

## Observed trained-arm gains

These are means over the ten final EMA E36–40 sweep summaries in each arm. The SD is the population SD over valid tile cells within a sweep. The changed fraction and Pearson correlation compare aligned and paired half-roll gains across those same positions. [All epoch and density gain statistics](refinement_gain.csv)

| Arm | Gain SD | Changed fraction | Mean paired Pearson |
| --- | ---: | ---: | ---: |
| S-guide | .030633 | 98.6878% | −.087859 |
| U-guide | .030816 | 98.6980% | −.096099 |
| S-placebo | .030559 | 98.6962% | −.089143 |

The permutation consistently changed the spatial assignment of gains in the trained-arm maps. The experiment did not establish a joint accuracy benefit for alignment or a sharing benefit. The six runs and the historical controls use the same internal validation split, with no independent replication.

## Evidence files

The [epoch index](epoch_index.csv) covers 18 tables with 84,720 rows. These contain epoch training and EMA metadata, live and EMA stage metrics, and per-image count, matching and cancellation measurements. The [resource audit](resource_audit.md) documents budget and background process observations. Large local logs and checkpoint binaries are referenced by path, byte size and SHA-256 in the run evidence.
