# LC-007B proposed preregistration

**Draft for owner review. No LC-007B execution or budget has been approved in LC-007A.** Treat this protocol as a single intervention with a fixed stopping rule. A separately approved preregistration must freeze code, local resource checks, seed queue and metrics before any updates.

## Intervention and common recipe

Parameter EMA with a fixed decay of .99 is the only proposed stabilization. Initialize the FP32 shadow by copying each live parameter before update one. After each successful AdamW update, update shadow in place as `shadow = .99 × shadow + .01 × live`. Do not advance on a skipped step. Evaluate live parameters and then shadow parameters at each scheduled validation epoch on the same images and saved **live** BN buffers, using eval mode. The shadow has no optimizer state, and loading it for inference must preserve exact live parameters and buffers before the next epoch. Preflight must check unchanged training loss/gradient/parameter and crop traces with shadow on/off, exact swap and restoration, finite shadow values and deterministic synthetic results. Keep the immutable live and shadow E40 states, epoch logs and config digests. Do not average BN buffers or fit thresholds.

Use the archived A train270/val30 and B train360/val40 manifests and exact label policy. Use the verified existing local ResNet-18 pretrained cache for each freshly initialized run. Seed common modules and first blocks identically across applicable arms and draw the second independent block separately. Use native 512 random crops, batch 2, overlap 128, BF16, train-mode backbone BN, stride 4 and K=2, same point loss (no density), AdamW weight decay .01, fixed 40 epochs and no early stopping. Use a 5% warmup, peak LR .0002 and cosine to .00002, with A 135 updates/epoch (5,400 total) and B 180 (7,200 total). Compare actual crop and LR digests per seed. Reuse the frozen .30 count cutoff, .05 AP floor, ownership, matcher and development splits. Record early/final for staged arms in one tile sweep per parameter state with no additional thresholds or selection. For late-epoch diagnostics retain per-image count/match summaries.

| Arm | Recurrent blocks / calls | Training supervision | Live and EMA readouts |
| --- | --- | --- | --- |
| CNN | 0 / 0 | Final | Final |
| Shared-1 | 1 / 1 | Final | Final |
| Shared-2 λ0 | 1 / 2 | Final + 0 × early | Early / final |
| Shared-2 λ.25 | 1 / 2 | Final + .25 × early | Early / final |
| Unshared-2 λ0 | 2 / 2 | Final + 0 × early | Early / final |
| Unshared-2 λ.25 | 2 / 2 | Final + .25 × early | Early / final |

Shared-1 and Shared-2 are the same-parameter comparison (1,596,614 measured parameters). Shared-2 and Unshared-2 are the same-call-depth and same-supervision comparison (historical fixed-window 32.745 GFLOP/tile, 1,596,614 versus 2,041,478 parameters). CNN is a common-backbone accuracy reference with 1,151,750 parameters. Report matched synchronized latency and allocated/reserved memory separately from approximate FLOPs. [Historical timing](../LC-004A/status.md)

## Stage 1 / fixed screening decision

Use B seeds **42 and 123**, each with CNN, Shared-2 λ.25 and Unshared-2 λ.25. The seeds span the stable-at-42 and unstable-at-123 recurrent histories, and CNN was unstable in both. Pair live and EMA inside each trajectory, six runs total. Order 42 as CNN → Shared-2 aux → Unshared-2 aux, 123 in reverse. Freeze one decay. Re-use live/EMA shared BN readouts every epoch, including E40, and derive stability with final E36–40 MAE population CV ≤10%, absolute E31–35 to E36–40 final-loss change ≤3% and finite logged losses, gradients and inference. The live and EMA readouts share the same training loss.

Advance for owner approval only if every run finishes the prespecified steps and all six EMA readouts meet stability, the EMA CV is lower than live in at least four of six, and each arm's two-seed mean EMA late MAE is at most 1.03 times its live mean with local nAP delta ≥−.005. Report both outcomes for all runs, including pair differences and FP/FN. Screen success only licenses consideration of the next stage. Any provenance mismatch, unintended download, missing labels, nonfinite state, missing step, resource or budget overrun stops the queue. Any failed screen criterion ends the proposed experiment at Stage 1. Preserve measured results and propose no decay sweep or additional screened seeds.

## Stage 2 / full fixed matrix

On separately approved expansion, use seeds **42, 123, 2026, 31415, 27182** in both A and B, six arms per seed, identical recipe and paired live/EMA output. Use the table order as the starting order, rotating left by the seed's index in the list and reversing odd-index rows. Stage 1 runs can count toward B only if the complete prefreeze config, implementation, data, budget, successful steps and live/EMA metrics match exactly. Keep their actual recorded order. Use EMA as the prespecified full-matrix primary readout and report live alongside it for every arm.

Use the per-seed arithmetic E36–40 mean MAE as primary, with RMSE, local nAP, signed counts and count-mask FP/FN reported separately. In each dataset require Shared-2 λ.25 to improve the five-seed mean late MAE by ≥3% versus CNN and independent Shared-1, with improvement in at least three of five paired seeds for each. Also report the ≥3% matched Unshared-2 λ.25 challenge, using the same direction rule and judgment on both A and B. Call the A/B shared architecture advantage supported only if the matched challenge and both other comparators meet the prespecified criteria in each dataset. Record any discordance. Report λ0/λ.25 and within-checkpoint early/final results separately. Preserve all per-seed differences and sample SDs.

For each comparator, use local nAP difference ≥−.005 as an exploratory tolerance and require an actual positive mean difference to claim a joint count and localization improvement. Guard pooled conclusions with mean late RMSE ≤1.03 times the comparator and each of mean FP and FN ≤1.05 times the comparator (absolute non-increase if a comparator mean is zero). All main arm/seed EMA readouts must satisfy the same stability rule, with finite validated metrics and no budget violation. These prospective rules leave LC-005/006 outcomes intact. Report failures in full and stop after the fixed budget. No statistical non-inferiority or significance claim follows from these descriptive checks.

## Compute envelope

Use the [per-run historical epoch timing](historical_recomputation.csv) on RTX 3090. Estimate the active training and validation queue by adding an extra full-validation estimate `epoch_seconds - train_seconds` to each historical full-epoch runtime. The difference includes non-training overhead and is a proxy for the second parameter readout. For unseen staged A Unshared-2 use the pooled LC-005 Shared-2 two-decode timing. The estimates exclude new EMA update and swap costs, extra per-image logging/checkpoint I/O, and contention beyond that observed.

| Phase | Trajectories | Updates | RTX 3090 active time estimate |
| --- | ---: | ---: | --- |
| B screen | 6 × 40 epochs | 43,200 | 2,936 s / 0.82 h. Historical original passes 1,998 s, estimated extra validation 937 s |
| Full A | 30 × 40 epochs | 162,000 | ~4.15 h (historical per-arm range projection 3.90–4.35 h) |
| Full B | 30 × 40 epochs | 216,000 | ~4.17 h (3.96–4.35 h) |
| Full A and B | 60 × 40 epochs | 378,000 | ~8.32 h (7.86–8.70 h) |
| Remaining with exact Stage 1 reuse | 54 × 40 epochs | 334,800 | ~7.5 h incremental |

Historical observed max training allocated/reserved were 615,471,616 / 805,306,368 bytes for staged A and 621,727,232 / 815,792,128 for B. The largest FP32 parameter shadow adds 8,165,912 bytes (7.8 MiB), with additional temporary swap storage and allocator uncertainty. For planning, propose a stop if any run exceeds 2 GiB process peak reserved, Stage 1 exceeds 1.25 active GPU hours, or the total reaches 11 active hours. Confirm the caps in a separated budget approval and measure actual pressure during a permitted future preflight before proceeding.

The existing A/B val populations are development evidence. Once model, threshold and protocol are frozen, a separate one-pass independent official test evaluation can be proposed and requires its own approval. Local nAP currently lacks official parity. P2PNet lacks a complete benchmark and NWPU is unavailable for this comparison. This draft authorizes neither source changes for EMA nor execution, evaluation or downloads.
