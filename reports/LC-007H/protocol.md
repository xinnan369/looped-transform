# LC-007H protocol

The canonical base is reviewed PR #30 head `4a0f7b023e314af8ac6fe3c8ea0390d3428dee63`. The formal control remains ungated Shared-2 aux+EMA. S4-shift is the research candidate. Use the fixed configurations and decision gates recorded in [the proposal](../../configs/lc007h_confidence_refinement.json) and [Issue #31](https://github.com/xinnan369/looped-transform/issues/31).

## Ordered stages

1. Reconcile archived per-image counts and match totals, verify opt-in identity parity on synthetic inputs, and freeze the guided and placebo variants on CPU.
2. After separate authorization, inspect four frozen LC-007G E40 EMA checkpoints in the B internal validation set. Complete [the resource plan](resource_plan.md) and the parity checks before interpreting association. Review all seed and density strata and report counts and empty groups.
3. Consider the six fresh training runs after review of the checkpoint diagnostic and a separate GPU authorization. Match LC-007G B train360/internal-val40 with 44,856/4,295 GT, native 512 crops, batch 2, overlap 128, stride 4, two slots, BF16, train BN, AdamW .0002 and .01 weight decay, 360-step warmup and cosine to .00002. Use FP32 .99 EMA over unique parameters with live BN, 40 epochs and 180 successful updates per seed/arm. Use one four-read sweep per epoch for live weights and one for EMA. Preserve fixed crop order and LR, owner count >.30 and AP floor >.05, count δ=.50, pooled local nAP δ=.05:.05:.50. Report E40 and live results separately from the primary equal-seed EMA E36–40 macro. Log staged TP/FP/FN and cancellation, state/update/gain statistics, and the density-stratified identity results when parity permits. Historical shifted controls are non-contemporaneous. Reconcile their initialization, actual crop/LR, EMA, metric and source hashes before comparison.

## Predeclared decision gates

For each of S-guide versus S4-shift, U-guide versus U4-shift, S-guide versus S-placebo, and S-guide versus U-guide, require ≥3% macro MAE reduction and strictly higher macro nAP, with both seeds favorable in both metrics. Per seed, sample MAE CV over E36–40 must be ≤10%, and absolute relative change in mean final loss from E31–35 to E36–40 must be ≤3%. Macro RMSE must be ≤1.03× control, and FP and FN must each be ≤1.05× control.

The matched S-guide threshold is MAE ≤21.98505 and nAP >.305224300. For U-guide versus U4-shift the corresponding limits are ≤20.1663 and >.305681402. Also report S2 parity ≤21.9375, S2 3% endpoint ≤21.279375 with nAP >.295320775, U2 3% endpoint ≤20.231775 with nAP >.300470175, and archived U4, U4-shift, CNN and Shared-1 comparisons. Keep thresholds, epochs, gain coefficient and stage selection fixed.

A matched-baseline improvement supports the candidate architecture in B. Spatial specificity requires passing the placebo gate. A sharing advantage requires passing the unshared guided comparison. If the joint matched gate fails, retain Shared-2 as the formal baseline and record the result without automatic retuning. Even a positive two-seed B comparison remains exploratory and requires independent A/B five-seed validation before generalization claims.
