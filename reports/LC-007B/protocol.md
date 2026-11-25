# LC-007B Stage 1 frozen protocol

Issue #19 authorizes eight fresh ShanghaiTech B runs on one RTX 3090, 57,600 successful experiment updates, 5,400 cumulative GPU-phase seconds, 2 GiB process peak reserved memory and 0 external download bytes. PR #18 head `a2df15cc8f817e8d8f5b1ab2273a8f312ab9a651` is the stacked parent. LC-007A keeps its original proposal as historical evidence.

## Queue and scoring

Reuse the four named LC-006 recipe files, the local ResNet-18 initialization and the B train360/internal-val40 manifest. Each run uses the same batch-two native random 512 px crops, 128 px overlap, stride 4/K2, BF16 and train-mode BN. AdamW uses peak LR .0002, WD .01, 360-update warmup and cosine to .00002 over 7,200 updates. Freeze 40 epochs, with no early stop or best-checkpoint selection. The two-call arms use the shared point decoder and final loss plus .25 × early loss. CNN and independently trained Shared-1 use final loss only. Disable shifting, loop embeddings, gains and density loss. Verify common and first-block initialization, an independent second unshared block, parameter counts, actual crop origins and applied LR hashes.

Seed 42 runs CNN, Shared-1, Shared-2 aux, Unshared-2 aux in order. Seed 123 runs them in reverse. Each trajectory starts from the cached initialization and produces both readouts within each validation epoch. Update a unique-parameter FP32 shadow at decay .99 immediately after each successful optimizer update. Use the same live BN buffers for live and EMA validation, with exact parameter, mode and RNG restoration. Incomplete runs retain the latest complete checkpoint and an interrupted snapshot when possible. Never restart an interrupted experiment run in this queue.

Each policy uses one tile sweep per epoch, with a single staged forward per tile for two outputs where applicable. Strict score >.30 and midpoint ownership define count. Local nAP pools one-to-one ranked matches over the original in-frame score >.05 pool. Count TP/FP/FN uses strict spacing-normalized δ .50. Use train-only linear density tertiles with lower-inclusive mid/high bounds. Group nAP pools matching ranks within the fixed stratum. Internal validation is development evidence. Record per-image count, signed and absolute error, GT and TP/FP/FN for every epoch/policy/stage.

## Frozen decisions

Primary is the E36–40 mean final count MAE, with epoch-40 results secondary. Stabilization requires all eight complete/finite, no resource breach, all EMA late MAE population CV ≤10%, all shared final-train-loss absolute relative E31–35 to E36–40 changes ≤3%, lower EMA CV than live in at least six runs, and each arm's two-seed mean EMA late MAE ≤1.03× live and local nAP change ≥−.005. Document both policies, each seed and RMSE, signed count and FP/FN. Compare Shared-2 aux with CNN, Shared-1 and Unshared-2 aux under each identical readout policy, then measure the EMA-minus-live change of that gap. High/mid/low comparisons and paired image data are descriptive. A passed stabilization screen would only support a future proposal. Stage 2 and independent test evaluation require separate authorization.

## Resource measurement and command

Synchronized elapsed phases include synthetic GPU preflight/profile, model setup, training, live validation, EMA validation, checkpoint transfer and serialization. Host work inside a phase is counted conservatively. Check the remaining time before each phase, training batch and validation image. The guard also checks the time and process allocator peak after completed operations. Use conservative upcoming-operation reserves with observed phase timing where available. A budget, finite-value, provenance, missing update or execution failure terminates the queue. Do not retry, add a seed or change the recipe. Synthetic disposable steps are counted in preflight time and separately from project updates.

```bash
source artifacts/lc002-env.sh
python -m scripts.run_lc007b --output "$HOME/loopcrowd-lc007b"
python -m scripts.archive_lc007b --local "$HOME/loopcrowd-lc007b"
```

All commits use `[skip ci]`. The protocol uses existing local resources without dependency installation, mirror probes, NWPU/WiderPerson access or official test access.
