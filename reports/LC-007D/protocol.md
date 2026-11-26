# LC-007D preregistration

**DRAFT_ARCHITECTURE / GPU_TRAINING_NOT_AUTHORIZED.** This is a two-seed exploratory screen on ShanghaiTech Part B. The matrix, guard values and tolerances are frozen in [the screen configuration](../../configs/lc007d_channel_screen.json). The [CPU auditor](../../scripts/prepare_lc007d.py) checks eligibility using the existing local cache. The [deferred runner](../../scripts/run_lc007d.py) requires separate, traceable owner authorization for the exact review commit, config digest and resource limits.

## Fixed experimental design

| Seed | Fresh order | Per-arm horizon |
| --- | --- | --- |
| 42 | R, RS, U-RS | 40 epochs × 180 successful updates |
| 123 | U-RS, RS, R | 40 epochs × 180 successful updates |

Both seeds use the LC-007B train360/internal-val40 B manifest with 44,856/4,295 cleaned labels, the verified ResNet18 `IMAGENET1K_V1` cache, fresh seed-specific initialization and training BN. The same two slots, native random 512 crops, batch size two, overlap 128, BF16, AdamW peak LR 0.0002, minimum cosine LR 0.00002, 360 warmup updates, weight decay 0.01 and final loss plus 0.25 early loss apply throughout. The EMA uses FP32 shadows with decay 0.99, updated once per successful optimizer step for each unique parameter, with live BN buffers. There are 240 epochs and 43,200 successful updates. Each run has the full horizon and both readouts. The protocol uses fixed thresholds and the final five epochs.

The eight historical LC-007B trajectories are fixed references from an earlier run. Within each seed the auditor reconciles the original experiment config, model initialization, manifest, local cache, actual archived crop and LR hashes, step counts, E36–40 image IDs, density bins, grouped count reconciliation and metric definitions. Each new run must then match its planned initialization at entry and its reference crop and LR digests after each epoch. Audit or trajectory mismatch stops the screen. No baseline reruns are in the queue. [Historical archive](../LC-007B/epoch_index.csv)

## Readouts

At both stages collect live and EMA count MAE, RMSE, signed and absolute count error, predictions, GT and TP/FP/FN, grouped over all data and frozen density bins. Group membership uses the LC-007B train-only tertiles 82.65177408854167 and 180.98619249131943 heads/Mpx. Validation uses the same forward for the early and final decoder. Count uses midpoint ownership with strict score >0.30. Local nAP uses the existing raw candidates above 0.05 and the existing local matching protocol with δ=0.50. nAP is pooled across ranked candidates and total GT within a validation group, then averaged across epochs. Report TP/FP/FN and count errors per image and matching density group. Report per-policy, per-stage, per-branch scale min/max/mean/L2 and delta L2. [Stagewise analysis plan](stagewise_analysis.md)

E36–40 per-run summaries average epochs equally. Macro summaries average the two seed-level summaries equally. Report E40 separately. For a pair, favorable MAE means strictly lower and favorable nAP means strictly higher. For each pair that has a joint gate, require favorable directions for both metrics in both seeds, macro MAE ≤0.97×, macro local nAP delta >0, macro RMSE ≤1.03×, and macro FP and FN each ≤1.05×. Per-seed EMA E36–40 MAE sample CV must be ≤10% and absolute relative change in mean final training loss from E31–35 to E36–40 must be ≤3% for the two arms in a joint gate. Undefined statistics and incomplete horizons leave the gate unevaluable. Training loss uses the fixed sampled crops. The primary endpoint is the final EMA readout.

1. **Primary** compares RS with ungated Shared-2 plus intermediate supervision.
2. **Residual ablation** compares R with that same reference under the joint gate. Report RS vs R directly with no additional pass threshold.
3. **Independent controls** compare R and RS with CNN and Shared-1. Report the 3% MAE and positive nAP targets. Compare each against the ungated Unshared-2 challenge MAE ≤20.231775 and nAP >0.300470 and report the paired directions.
4. **Matched sharing challenge** compares RS with U-RS under the joint gate, with matched gate shape and depth. The original nAP margin of −0.005 is exploratory and does not enter the joint decision.

This screen can guide selection for a separately preregistered and authorized A/B five-seed replication. Require at least 3/5 favorable paired directions in that follow-up with independent CNN, Shared-1, ungated Shared/Unshared and the matched gated Unshared control. Keep the existing Shared-2 + auxiliary loss + EMA configuration if neither shared gate reaches joint improvement. No post hoc threshold or checkpoint selection.

References for the design and scope are [Issue #23](https://github.com/xinnan369/looped-transform/issues/23), [LC-004A scalar gains](https://github.com/xinnan369/looped-transform/pull/10), [LC-007C Issue #21](https://github.com/xinnan369/looped-transform/issues/21), and [LC-007C draft PR #22](https://github.com/xinnan369/looped-transform/pull/22).
