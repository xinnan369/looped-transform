# LC-007B Stage 1 result

**The frozen stabilization screen passed.** All eight ShanghaiTech B runs finished 40 epochs with 7,200 successful updates each. The 57,600 project updates and 48 disposable synthetic preflight steps used 4,043.21 synchronized GPU-phase seconds (67.39 minutes). The process peak reserved memory was 713,031,680 bytes (680 MiB). The queue stayed within both caps. [Execution and decision](research_decision.json) · [GPU phases](gpu_phases.csv)

Every EMA final readout met the 10% E36–40 MAE CV limit and had strictly lower CV than its own live readout. Two live readouts met that limit. The EMA CV range was 2.51–7.08%. All eight absolute E31–35 to E36–40 final training loss changes were at most 2.27%. Each arm met the two-seed MAE and nAP guards, and the observed E36–40 mean MAE decreased in every individual run.

| Arm | Live → EMA late MAE | Live → EMA RMSE | Live → EMA local nAP | Live → EMA mean CV |
| --- | ---: | ---: | ---: | ---: |
| CNN | 29.2375 → 27.2125 | 39.416 → 36.663 | .274616 → .275499 | 11.88% → 5.99% |
| Shared-1 | 25.3700 → 22.5150 | 33.470 → 29.671 | .289021 → .290406 | 13.39% → 6.63% |
| Shared-2 aux | 25.2050 → 21.9375 | 33.620 → 29.278 | .293146 → .295321 | 11.55% → 4.56% |
| Unshared-2 aux | 24.6800 → 20.8575 | 33.303 → 27.660 | .297971 → .300470 | 11.07% → 3.26% |

Each row equally weights the two seeds. CV is computed per run using the population SD of its five late MAEs. See the [experiment matrix](experiment_matrix.csv) for individual seeds, FP/FN, counts, loss changes and checkpoint SHA values. The [paired policy table](paired_live_ema.csv) separates E36–40 and supplemental E40 results and includes early readouts where applicable.

## Architecture comparison under the same EMA policy

| Shared-2 aux vs comparator | ΔMAE seed 42 / 123 | Mean ΔMAE | Relative MAE change | Mean Δ local nAP |
| --- | ---: | ---: | ---: | ---: |
| CNN | −7.330 / −3.220 | −5.2750 | −19.38% | +.01982 |
| Independent Shared-1 | +0.700 / −1.855 | −0.5775 | −2.57% | +.00492 |
| Matched Unshared-2 aux | +2.700 / −0.540 | +1.0800 | +5.18% | −.00515 |

Under EMA, Shared-2 aux beat CNN in both seeds. Its gain versus Shared-1 fell short of the separately proposed 3% target and changed direction across seeds. Matched Unshared-2 aux had the lowest two-seed MAE and the highest local nAP. Shared-2 aux improved its own MAE in all three strata, with an 8.24-head reduction in the high-density stratum and 27.8 additional FP / 55.7 fewer FN overall. Against matched Unshared-2 aux, the high-density MAE deficit grew from +0.60 live to +1.77 EMA. Its high-density advantages versus CNN and Shared-1 grew by 2.63 and 0.95 heads, respectively. [Architecture comparisons](architecture_comparison.csv) · [Density analysis and paired images](density_analysis.md)

The live and EMA readouts share one training trajectory and the same live BN buffers per epoch. These two B development-validation seeds support the stabilization screen. A recurrent architecture accuracy advantage and official-test performance remain unestablished.

## Evidence

- [Protocol](protocol.md) and [preflight](preflight.json) pin the dataset, cached weight, queue, phase budgets, seeds and code freeze `6471f222dbe1480ff7fca1ef0291215f2498d32f`.
- [Epoch and image index](epoch_index.csv) lists 24 split CSVs with SHA-256, byte counts and row counts. The data include 320 training epochs, 640 full validation sweeps and 38,400 image/readout rows.
- [Local verification](test_results.md) records three full 103-pass suites and exact LC-006 final live metric and loss agreement across 320 matching epochs. [Resource audit](download_audit.md)

Stage 1 is ready for owner acceptance on the stacked branch. Stage 2 and official testing require their own approval. [Issue #19](https://github.com/xinnan369/looped-transform/issues/19) · [Next steps](next_steps.md)
