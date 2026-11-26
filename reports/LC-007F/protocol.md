# LC-007F eight-call screen

**DRAFT_DEPTH8 / GPU_TRAINING_NOT_AUTHORIZED.** Test fresh K=8 starts after the completed negative K=4 factorial. All four K=4 arms had higher macro MAE than corresponding K=2 arms. S4 improved local nAP from stage 3 to stage 4 while count MAE worsened in both seeds. The retained control is ungated Shared-2 with auxiliary supervision and EMA. K=8 tests a bounded depth hypothesis in the fixed-window model inspired by Déjà View. [Issue #27](https://github.com/xinnan369/looped-transform/issues/27) · [K=4 report](../LC-007E/status.md)

| Seed | Serial order | Horizon per arm |
| --- | --- | --- |
| 42 | S8, U8, S8RS, U8RS | 40 × 180 successful updates |
| 123 | U8RS, S8RS, U8, S8 | 40 × 180 successful updates |

Use the verified ShanghaiTech B manifest with 360 training and 40 internal-validation images and 44,856 and 4,295 cleaned heads. Use the same local ResNet18 checkpoint and fixed seed starts. Batch 2, BF16, native random 512 crops, 128 overlap, stride 4, two slots, training BN, AdamW peak LR 0.0002 and weight decay 0.01, 360 warmup updates and cosine minimum LR 0.00002 match the historical recipe. FP32 EMA uses decay 0.99 on unique parameters with live BN buffers. The queue contains 320 epochs and 57,600 successful updates.

Training uses exactly two decoder invocations, at calls 4 and 8, with final point loss plus 0.25 midpoint point loss. Relative supervisory positions match K=2/4, while absolute position and compute differ. Runs use fresh initializations at their assigned depths. Compare only after historical K=2/4 manifests, cache, first/common/decoder digests, inherited block digests, source and metrics, actual crop/LR trajectories, optimizer and EMA steps are reconciled.

For epochs 1–35 validation measures calls 4 and 8 under live and EMA. For epochs 36–40 it measures calls 1–8 under each of the same two existing sweeps. This differs from K=4's four reads every epoch. Freeze score >0.30 with midpoint ownership for count and raw score >0.05 for AP. Density boundaries are 82.65177408854167 and 180.98619249131943 heads/Mpx from the training split. Count matching uses strict normalized δ=0.50. Local nAP averages pooled AP at δ=0.05 through 0.50. Stable score ties preserve candidate order and GT ties use normalized distance then GT ID. Pooling preserves validation image order. [Arm recipe](../../configs/lc007f_depth8.json) · [Plan and tolerances](../../configs/lc007f_depth_screen.json)

## Decision rules

Use final EMA E36–40 as primary, average epochs within seed and then weight seeds equally. Report E40 and live as supplementary. Compare S8 with S4, S2 and U8. Compare S8RS with S4RS, S2RS and U8RS. Within K=8 compare S8RS vs S8 and U8RS vs U8. Each joint test requires macro MAE ≤0.97× the reference and strictly higher local nAP, favorable MAE and nAP in both seeds, late sample MAE CV ≤0.10 and absolute E31–35 to E36–40 sampled final loss change ≤0.03 in each compared arm and seed. Require macro RMSE ≤1.03× and FP and FN each ≤1.05×. Missing horizons and undefined guards leave the joint test unevaluable.

Report the independent CNN, Shared-1 and ungated Unshared-2 3% MAE and higher-nAP targets. Compare R descriptively. For each gate stratum calculate (S8−U8)−(S4−U4) and (S8−U8)−(S2−U2) using the matched references. Lower MAE and higher nAP interactions indicate narrowing in the respective metrics. Direct jointly guarded tests determine the sharing claim. Select no prefix, threshold or epoch after observing outcomes. Repeated epochs and images within the two-seed Part B screen do not supply independent replications. A positive result supports a separately reviewed A/B five-seed proposal. An unsuccessful result retains the Shared-2 default and prompts a review of the absolute MAE trend.

[Issue #25](https://github.com/xinnan369/looped-transform/issues/25) · [PR #26](https://github.com/xinnan369/looped-transform/pull/26) · [Issue #23](https://github.com/xinnan369/looped-transform/issues/23) · [PR #24](https://github.com/xinnan369/looped-transform/pull/24)
