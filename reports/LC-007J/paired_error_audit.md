# Where the additional false positives appeared

Both archives cover the same 40 internal-validation images in identical order across every epoch, stage and policy. The audit checked exact indexed file sizes and SHA-256, source commits and E40 checkpoint provenance, GT and density identity, grouped counts, and `FP + FN = |FP − FN| + 2 min(FP,FN)` for every image. Precision is TP divided by owned predictions and recall is TP divided by all GT. Reported late rates are means of per-epoch ratios, weighting E36–40 equally. These are repeated measurements on the same images. Ranked nAP is carried from the verified grouped archive. [LC-007G source index](../LC-007G/epoch_index.csv) · [LC-007I source index](../LC-007I/training_results/epoch_index.csv) · [Source and checkpoint digests](evidence_summary.json)

## Final EMA E36–40

| Metric | Native 512 | Native 768 |
| --- | ---: | ---: |
| MAE | 23.875 | 19.700 |
| RMSE | 31.815 | 25.968 |
| Archived local nAP | .305537 | .346196 |
| TP / FP / FN per sweep | 2,552.6 / 1,351.0 / 1,742.4 | 3,066.8 / 1,760.6 / 1,228.2 |
| Precision / recall | 65.39% / 59.43% | 63.54% / 71.40% |
| Signed count error | −391.4 | +532.4 |
| Under / exact / overcounted images | 25.0 / 0.4 / 14.6 | 7.6 / 1.8 / 30.6 |
| Total under / overcount magnitude | 673.2 / 281.8 | 127.8 / 660.2 |
| Canceled unmatched events | 2,138.4 | 2,200.8 |

Additional FP account for +409.6 of the +923.8 signed-error shift and the reduction in FN accounts for +514.2. Absolute count error fell by 167 per sweep. Total unmatched events fell by 104.6 and canceled events increased by 62.4. The greater recall and higher ranked nAP show improved localization on these controls. Cancellation also contributed to the MAE reduction.

FP increased in 33 of 40 images by late-epoch mean. The gross increase was +477.8, offset by −68.2 in seven images. The five largest increases contributed 35.50% of gross positive change. The largest was +54.8 at `train/IMG_186`. Low, middle and high density contributed +90.0, +147.4 and +172.2 FP respectively. All three strata changed from aggregate undercounting to overcounting. This indicates both a broad rise and some image concentration. [Paired image rows](paired_image_error.csv) · [Per-epoch and density records](stage_density_summary.csv)

## Readouts and secondary endpoints

For 768 final EMA by stage, t2→t3→t4 MAE was 18.54→22.48→19.70 and archived nAP rose .338860→.345021→.346196. The final step removed 143 FP and added 39.4 FN on average, moving the signed error from +714.8 to +532.4. These counts do not establish that the same real heads were lost or that suppressed predictions were duplicates.

At E40 EMA, 512 recorded 23.025 MAE and .307819 nAP, and 768 recorded 20.325 and .345710. E36–40 live final was 24.260/.302793 for 512 and 21.465/.345479 for 768. E40 live was 24.175/.299401 and 24.475/.344384 respectively. These checks are secondary. The original joint fail, thresholds and E36–40 primary endpoint remain in force. [Per-epoch and stage summary](stage_density_summary.csv)

The training artifacts retain no candidate or GT matching export for the 768 run. The earlier H read-only 512 E40 export for the exact same checkpoint contains midpoint candidates and GT transitions. This supplies a partial 512 identity record only. Paired candidate attribution, duplicate versus background FP, tile seams and AP reconstruction remain unavailable from the paired training archives. [Identity inventory](evidence_summary.json)
