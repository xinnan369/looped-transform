# E40 paired error attribution

All 33 frozen epoch-40 checkpoints completed one internal-validation pass, yielding 51 readouts. The 51 comparisons in [E40 reconciliation](e40_reconciliation.csv) reproduce the available archived count, local nAP and match metrics. A has 30 validation images and 14,177 GT heads, with 1–9 tiles per frame and native dimensions of 432–1,024 × 267–1,024 px. B has 40 validation images and 4,295 heads, all 1,024 × 768 px and six tiles each. The table reports the median of image-wise median nearest-neighbor spacings.

| Dataset | Train-only tertiles, heads/Mpx | Group | Val images | GT | Nearest GT, px |
| --- | --- | --- | ---: | ---: | ---: |
| A | 545.595 / 1,080.578 | Low | 11 | 1,996 | 21.36 |
| A | | Mid | 8 | 3,618 | 14.81 |
| A | | High | 11 | 8,563 | 8.76 |
| B | 82.652 / 180.986 | Low | 13 | 566 | 36.49 |
| B | | Mid | 16 | 1,511 | 21.35 |
| B | | High | 11 | 2,218 | 11.73 |

[Stage and density metrics](e40_groups.csv) contain exact pooled local nAP within each group, counts and the top-five IDs. [Paired summaries](paired_summary.csv) give group-specific MAE, RMSE, nAP and FP/FN, seed-level results and the fractions of paired images that improve, worsen or tie. The [image index](paired_image_metrics.csv) has per-stage and paired file paths, checksums and status. Each stage CSV carries image geometry, histograms and match and coverage diagnostics without candidate or GT coordinates. Group AP pools matches inside the group and whole-validation AP pools all images.

## Independent training comparisons

The table gives the three-seed mean target-minus-reference difference. An MAE delta below zero favors the target. A comparisons with LC-003 cross archives and the A matched Unshared-2 λ.25 cell is unavailable.

| Part | Target vs reference | ΔMAE all / low / mid / high | ΔRMSE all | Δ local nAP all | ΔFP / ΔFN all |
| --- | --- | --- | ---: | ---: | ---: |
| A | Shared-2 aux vs λ0 | −15.00 / −6.36 / −18.33 / −21.21 | −20.72 | −.00064 | −298.3 / +25.0 |
| A | Shared-2 aux vs CNN | +9.22 / −7.73 / +24.63 / +14.97 | +29.08 | −.00282 | −61.7 / −211.7 |
| A | Shared-2 aux vs Shared-1 | −2.74 / +1.39 / −0.13 / −8.79 | −1.65 | −.00397 | −436.7 / +219.0 |
| B | Shared-2 aux vs λ0 | −1.18 / −0.38 / −0.85 / −2.61 | −1.57 | +.00014 | +19.7 / −27.7 |
| B | Shared-2 aux vs CNN | −1.73 / −2.79 / −0.90 / −1.70 | −1.04 | +.01631 | +251.3 / −333.3 |
| B | Shared-2 aux vs Shared-1 | +3.94 / +0.05 / +2.54 / +10.58 | +6.36 | +.00153 | −70.0 / +29.7 |
| B | Shared-2 aux vs Unshared-2 aux | +3.02 / +0.74 / +1.33 / +8.15 | +3.93 | −.00426 | −96.7 / +124.0 |
| B | Unshared-2 aux vs λ0 | −1.14 / −0.97 / −0.69 / −2.00 | −1.14 | +.00176 | +46.7 / −75.0 |

High density had the largest mean count gain from shared supervision on both datasets. High-density local nAP nonetheless fell .00360 on A and .00155 on B versus shared λ0. The high-density B deficit was largest versus both independent controls. It accounted for 74% of the total paired MAE disadvantage versus Shared-1 and 74% versus same-depth Unshared-2 aux when each group's MAE delta is multiplied by its image count. Against Shared-1, the B shared aux improved 35.8% of images, worsened 58.3% and tied 5.8% averaged across seeds. The corresponding percentages against Unshared-2 aux were 37.5%, 60.8% and 1.7%. A showed a smaller high-density MAE versus its historical Shared-1, with overall nAP lower and an A/B sign change in the count comparison.

## Within-checkpoint readouts

The A supervised final changed MAE by −0.32, RMSE by +6.97 and local nAP by +.00022 versus its own early readout. It worsened high-density MAE by 20.36 and nAP by .00645 while improving mean count MAE in the other two groups. The B supervised final changed MAE by −0.71, RMSE by −1.20 and local nAP by +.00766, with high group MAE −2.21. The seed-42 E40 B MAE delta was +2.18. The archived B late-epoch improvement in every seed addresses a different endpoint. B Unshared-2 aux also improved local nAP at the second call (+.01117), with a small higher MAE (+0.37). Large final-over-early gains in λ0 cases primarily compare a supervised output with a prefix that had zero direct loss.

## Candidate coverage and spatial accuracy

For Shared-2 aux final, raw-floor proximity covers 99.83% of A and 99.43% of B validation GT in the three-seed means. Ownership alone lowers those figures to 99.78% and 99.38%. The owned count view covers 59.19% and 61.41%. The corresponding fixed one-to-one recalls are 57.77% and 58.32%. Permissive proximity can count one candidate for several GT and benefits from large low-ranked pools. Its small raw-to-owned change and large owned-to-count change locate most of the observed coverage loss at acceptance of high scores. This diagnostic leaves open whether loss optimization, ranking, tiling context or localization caused individual misses. Earlier LC-004B held the candidate pool fixed and its training-fit calibration missed the joint comparator gate.

On B, Shared-2 aux increased early-to-final count-mask proximity from 58.53% to 61.41% and gained 116 TP with 88.3 more FP per seed. The TP-weighted accepted distance changed from 3.303 to 3.408 px (normalized .12863 to .12870). The matched GT population changed between stages. The higher local nAP can coexist with greater accepted distances and depends on rank and recall relative to all GT. On A the corresponding accepted distances were 2.688 to 2.707 px (.20618 to .20533 normalized). On B, matched Unshared-2 aux had 63.91% count proximity and 124 more TPs than shared aux, with 96.7 more FPs. All above distances use accepted owned count pairs at strict δ .50.

Internal seams within 4 px contain 321/14,177 A labels and 94/4,295 B labels. Of Shared-2 aux's average final FN totals (5,988 A and 1,790 B), 160 and 35 fell in those bands. Seam failures can contribute, but the measured bands hold a small share of the total misses. The fixed-origin stride-four/K2 proxy overflow is zero on A validation and one on B. Earlier crop logs include overflow in densely populated training regions. This proxy does not enumerate tile-rebased slots or establish a model recall cap. Score histograms, ownership exclusions and ambiguity diagnostics are provided per image.

## Image sensitivity versus seed dispersion

The largest five images account for 51.5–63.5% of total absolute error per A final checkpoint and 31.1–38.0% per B final checkpoint. Their IDs are filed in the stage metrics. The following distributions average seeds inside each image resample. The seed sample SD instead uses three complete-validation means.

| Part and shared aux contrast | Full ΔMAE | Seed sample SD | Bootstrap p2.5 / median / p97.5 | Drop-one image range |
| --- | ---: | ---: | --- | --- |
| A vs shared λ0 | −15.00 | 25.32 | −24.47 / −14.49 / −6.28 | −17.00 to −12.26 |
| A vs CNN | +9.22 | 14.60 | −11.72 / +8.82 / +30.65 | +3.53 to +14.00 |
| A vs Shared-1 | −2.74 | 9.11 | −20.60 / −2.97 / +17.36 | −9.34 to +2.39 |
| B vs shared λ0 | −1.18 | 3.73 | −2.34 / −1.16 / −0.03 | −1.34 to −0.92 |
| B vs CNN | −1.73 | 3.96 | −5.85 / −1.77 / +3.09 | −3.01 to −0.97 |
| B vs Shared-1 | +3.94 | 0.58 | +1.77 / +3.90 / +6.26 | +3.31 to +4.37 |
| B vs Unshared-2 aux | +3.02 | 2.79 | +1.27 / +2.96 / +4.88 | +2.65 to +3.31 |

Omitting any single B image leaves the two independent comparator gaps positive. A's Shared-1 comparison can change direction under one omission. Bootstrap percentiles are descriptive for these development images and seeds. [Sampling evidence](sampling_sensitivity.json) includes the ordering, shared index digest, all omitted-image results and all seed deltas. The measurements provide no historical per-epoch image decomposition, population inference or test benchmark.
