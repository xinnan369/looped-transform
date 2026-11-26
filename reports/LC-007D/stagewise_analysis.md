# Stagewise measurements

The table shows final minus early changes for EMA E36–40, equally weighted across the two seeds. Pooled local nAP was calculated within each group and epoch before averaging. [Grouped corrections](stagewise_correction.csv) · [Per-image corrections](image_stagewise_correction.csv)

| Arm | Group | Δ MAE | Δ RMSE | Δ local nAP | Δ FP | Δ FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R | All | −2.875 | −3.773 | +0.009940 | +25.3 | −83.3 |
| RS | All | −2.758 | −3.276 | +0.010913 | −2.4 | −101.5 |
| U-RS | All | −1.158 | −1.960 | +0.014046 | −50.9 | −112.2 |
| R | High | −7.755 | −7.427 | +0.006524 | +30.8 | −44.7 |
| RS | High | −6.182 | −5.857 | +0.007332 | +10.9 | −48.3 |
| U-RS | High | −2.155 | −2.427 | +0.010979 | −26.6 | −41.3 |

All three arms reduced count error from early to final. U-RS gained more nAP than both shared gated arms in all-data and high-density macro summaries. On all data, the R change in matched count error included an increase of 57.0 canceled unmatched predictions and GT heads, averaged over the two seed windows. The corresponding changes for RS and U-RS were +6.4 and −116.8. Cancellation is computed per image as twice the smaller of FP and FN. [Gate scale statistics](gate_statistics.csv)

Candidate identity, GT-level recovered/lost/both/unmatched transitions, and matched-only displacement, score and rank changes are `not_available` in this design. The frozen inference path retains tile assignment and coordinates/scores but discards the cell and slot indices used by the decoder. No extra inference sweep is scheduled. Scale distributions describe what the model learned. Evaluating causal effects would require an intervention outside this protocol. [Readout and thresholds](protocol.md)
