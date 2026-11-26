# Approved LC-007C endpoint execution

On November 26, 2026 (Asia/Shanghai), the project owner explicitly approved the previously presented 96 fixed cases, 1,200 synchronized seconds, 2 GiB process peak reserved, zero project optimizer steps and zero external downloads, and asked to check background training first. Scope is the frozen [Issue #21](https://github.com/xinnan369/looped-transform/issues/21) diagnosis on the four existing E40 files. CPU draft `297401f0eba4f8ef90185b8c4db3b478be488a3e` preserves the original proposal and acceptance tolerances. The [probe manifest](probe_manifest.json) retains SHA-256 `aa9ca77267d849d60c9c767139306bdca8ed43aeb5015756f168ff215dc37957`.

## Concurrent workload

A separate training process (WSL PID 89303) was active at the initial check. At 08:31 CST the eight one-second samples showed about 19,074–19,107 MiB free, with GPU utilization 4–96%. A prior immediate sample showed 18,987 MiB free and 36% utilization. This leaves allocation headroom for the 2 GiB process cap. Device contention can affect wall timing. Preserve the other job and use synchronized measured times, including preflight and release. An allocation refusal, time projection failure, nonfinite result, failed equivalence, synchronization error or invariant failure stops the queue and retains evidence.

## Execution freeze

Reverify the exact local checkpoint, train-only manifest and ResNet18 cache hashes, and each saved checkpoint's code/config, final epoch, EMA, optimizer and scheduler position. Match the original fixed-loop BF16 training graph and point loss at λ=.25. Only one GPU process and one checkpoint resident at a time. The order follows the manifest, then ascending batch index, with live preceding EMA. Read the fixed batch once for the policy pair and verify the transformed data/target hash across arms. Keep the same checkpoint BN buffers for both readouts, fork RNG around the two stateless forward graphs and compare parameter/buffer identity, versions, model modes, gradients and RNG after each case. Verify exact model/EMA/optimizer/scheduler tensor or metadata hashes and checkpoint file SHA after each checkpoint, including an interrupted one. The runner does not instantiate an optimizer.

Run one generated batch-two 512 px BF16 preflight per architecture, charged as GPU work and separate from the 96 project cases. Use the already frozen closeness limits from the CPU draft. Finite outputs, losses and all reachable VJPs are required. Record legitimate absent gradients and undefined cosines without zero imputation. On any failed preflight or case, keep the first observed status and diagnostics and stop before the next project case. No threshold change, fallback precision or rerun is included.

The [budget guard](../../src/loopcrowd/probe_budget.py) accounts for initialization, transfers, synthetic checks, batch probes and release, including exceptional paths and final synchronization. Reserve 12 seconds for release and take twice the longest completed synthetic or case phase as the next-case estimate. Record actual cap crossings separately from a pre-entry forecast refusal. Mandatory release stays charged after a failure. The guard is cooperative and allocator-based.

## Reporting and interpretation

Keep full per-case observations by checkpoint, policy, seed, density, loss and parameter group. Summarize measured, undefined and no-path denominators and descriptive cosine/projected-norm distributions. Off-diagonal unshared-block gradients compare distinct tensors and have descriptive scope. Examine recurrence and supervision separately. Directional conflict that is common across seeds and appears in the dense group would be a candidate association with the smaller shared second-call nAP gains. Inconsistent seed/policy directions, small opposing projections or absent gradients limit that association. Static endpoint measurements cannot establish an optimization mechanism or justify an architecture intervention by themselves. Retain the early count gap and stagewise correction as separate evidence.

```bash
source artifacts/lc002-env.sh
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_lc007c \
  --approved-scope LC-007C-96-1200-2GiB-zero-updates-zero-downloads \
  --output "$HOME/loopcrowd-lc007c-gpu"
```

No validation sweep, Stage 2 or hosted CI. Keep the stacked PR unmerged and preserve the accepted historical commits.
