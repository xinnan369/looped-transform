# Stagewise error correction on frozen ShanghaiTech B validation

Checked the frozen LC-007B index and 24 referenced files, totaling 42,560 rows. Recomputed the complete E36–40 stage/policy/image join, including eight trajectories, 80 grouped paired comparison rows and 800 paired image rows for the two two-call arms. Group membership was 13 low-density images with 566 GT, 16 mid with 1,511 GT and 11 high with 2,218 GT. Validation tertiles are 82.6518 and 180.9862 heads/Mpx, derived from original training image density. [Integrity](archive_integrity.json) · [Grouped and macro calculations](archive_correction.csv) · [Per-image correction](archive_image_correction.csv)

## Absolute count error

The gap is Shared-2 aux minus same-supervision Unshared-2 aux. Positive values indicate higher shared-model error. The final gap is the early gap plus the difference between the two within-checkpoint corrections. All table entries equally weight E36–40 within seed and then the two seeds.

| Policy | Stratum | Early MAE gap | Shared early→final ΔMAE | Unshared early→final ΔMAE | Correction difference | Final MAE gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Live | All | +1.468 | −1.033 | −0.090 | −0.943 | +0.525 |
| Live | Low | −0.123 | +0.431 | +0.246 | +0.185 | +0.062 |
| Live | Mid | +1.100 | −1.288 | −1.038 | −0.250 | +0.850 |
| Live | High | +3.882 | −2.391 | +0.891 | −3.282 | +0.600 |
| EMA | All | +1.738 | −2.533 | −1.875 | −0.658 | +1.080 |
| EMA | Low | +0.131 | +0.177 | +0.677 | −0.500 | −0.369 |
| EMA | Mid | +1.238 | −1.750 | −2.294 | +0.544 | +1.781 |
| EMA | High | +4.364 | −6.873 | −4.282 | −2.591 | +1.773 |

The average high-density EMA count deficit is present at the first readout. The second call reduces it by 2.59 heads. The final high-density gap widened by 1.17 from live to EMA, decomposed into +0.48 at the first readout and +0.69 in the relative correction term. Within EMA, seed 42 widened the high-density gap from +5.69 to +7.07 at the second call. Seed 123 narrowed it from +3.04 to −3.53. The overall EMA relative correction also differed by seed (+0.47 and −1.79). These directions limit an explanation based on a universal count-refinement failure.

The EMA shared readout improved absolute error on 58.25% of paired image/epoch observations, tied on 10.00%, and worsened on 31.75%. The independent two-call control reached 56.75%, 6.75% and 36.50%. In the high group the shared rates were 78.18%, .91% and 20.91%. The independent rates were 70.00%, .91% and 29.09%. The low group saw mean MAE increase for both models despite many improved images. These fractions are descriptive and reuse the same validation IDs each epoch.

## Localization and count cancellation

Both EMA models improved pooled local nAP after call two. The gains were +.009748 shared and +.013134 unshared overall, and +.006762 / +.009513 at high density. The shared correction deficit in nAP was −.003386 overall and −.002751 at high density. Its direction held in both seeds and both policies for these strata. The macro EMA final nAP gap was −.005149 overall and −.004223 at high density. Early gaps were already −.001764 and −.001472. The localization deficit increases at the second readout while the macro count gap decreases. These are pooled AP values saved for each group/epoch, with no reconstruction from the count mask.

EMA overall RMSE changed from 32.95 to 29.28 for Shared-2 aux and 30.58 to 27.66 for Unshared-2 aux. At high density the changes were 49.60→42.36 and 45.59→41.01. The high-density final signed totals were −276.6 and −258.7 heads per validation epoch, with both models undercounting on net.

| EMA stratum / arm | ΔFP | ΔFN | Δ(FP+FN) | Δ2Σmin(FP,FN) | ΔMAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All / Shared | +18.2 | −106.7 | −88.5 | +12.8 | −2.533 |
| All / Unshared | −9.9 | −106.1 | −116.0 | −41.0 | −1.875 |
| High / Shared | +13.1 | −53.9 | −40.8 | +34.8 | −6.873 |
| High / Unshared | −1.2 | −40.3 | −41.5 | +5.6 | −4.282 |

At high density the unmatched-error totals fall by similar amounts, and the difference in how FP and FN offset one another drives the relative MAE correction. Overall, the shared model's final EMA readout still has +15.9 FP and +63.7 FN relative to Unshared-2 aux. The high-density differences are +11.2 FP and +29.1 FN. These are net changes in confusion counts at fixed scoring. The records do not retain pair identities for individual GT, so recovery and loss of specific heads remain unmeasured.

## Independently trained references

On EMA E36–40 final readouts the shared arm differed from independent Shared-1 by −0.578 MAE and +.00492 local nAP, and from CNN by −5.275 MAE and +.01982 local nAP. The MAE differences versus Shared-1 were +0.700 and −1.855 across the two seeds. The early prefix in the stage decomposition is from the two-call training run, with its own supervision and shared parameters. The independent baselines are kept separately in all rows. E40 is supplementary. For EMA E40 high density the early gap was +5.45, the correction difference −2.09 and the final gap +3.36.

The archive diagnoses the location of count and localization discrepancies. Gradient conflict, rank/localization distortion and model capacity remain competing explanations.
