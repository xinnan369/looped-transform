# Proposed K=8 resource envelope

**GPU_TRAINING_NOT_AUTHORIZED.** Proposed hard limits are a single RTX 3090, eight serial fresh runs, 320 epochs, 57,600 successful project updates, at most 10,800 charged seconds, at most 2 GiB peak process reserved, zero disposable optimizer updates and zero external experiment bytes. [Proposed limits](../../configs/lc007f_depth_screen.json)

## Projection method

| Phase | Projected seconds |
| --- | ---: |
| Training | 3,098 |
| Live validation | 2,395 |
| EMA validation | 2,356 |
| Checkpoints | 40 |
| Preflight, transfers and cleanup reserve | 120 |
| Central estimate | 8,009 |
| With 20% contingency | 9,611 |

The time proxy with contingency is about 160.2 minutes, 1,189 seconds under the proposed ceiling. The CPU graph retained 494,937,480 bytes at K=4 and 788,424,584 bytes at K=8. The derived process peak proxy is 1,649,496,165 bytes, about 1.54 GiB. Both projections fit the proposed envelope. Actual GPU feasibility remains pending.

Scale the measured LC-007E training sum by the ratio of K=8 and K=4 two-read dense forward FLOPs. Scale live and EMA validation separately by the epoch-weighted ratio of 35 K=8 two-read epochs and five K=8 eight-read epochs to 40 K=4 four-read epochs. Double checkpoint cost and add 120 seconds for preflight, setup, transfers and cleanup, then add a 20% contingency. CPU matching, kernels and contention can move actual times away from the approximation. [Historical ledger](../LC-007E/gpu_phases.jsonl) · [Projection data](initialization_audit.json)

The memory calculation uses a CPU BF16 differentiable synthetic forward at the full 512 tile and batch 2 for U4RS and U8RS. Count unique backing storages saved for the two-loss backward graph. Scale the actual LC-007E peak reserved by that ratio and add 20 bytes per additional unshared parameter plus 64 MiB. The 20-byte allowance covers weights, gradients, AdamW moments and EMA shadows. The result is a sizing proxy. GPU workspace allocation and allocator fragmentation are measured only after authorization. The largest variant has fewer than five million parameters. Eight retained checkpoints at 16 bytes per parameter contribute under 640 MB before metadata. The queue requires 4 GiB free for checkpoints, interruptions and evidence.

## Deferred execution controls

The documented authorization must match the reviewed code commit, exact plan digest and limits. The runner verifies cached inputs and stops before CUDA on projection failure. After separate authorization it records existing compute work and verifies at least 2 GiB available on one BF16-capable RTX 3090, then caps allocator fraction. Concurrent workloads continue. The charged latency probe uses generated BF16 tiles, rotates arm order and alternates final-only, midpoint/final and eight-read modes. It takes one warmup and five samples for each arm and mode with zero optimizer updates. Other-process contention is disclosed from the start and end snapshots. [Deferred runner](../../scripts/run_lc007f.py)

Use next-phase floors of 30 seconds for preflight, 20 for train, 25 for each validation policy and two for checkpoint, with observed maxima thereafter and a 20-second cleanup reserve. Charge synchronization, setup, transfers, interphase time, serialization, cleanup and failed work. Stop at an estimated overrun, OOM, nonfinite, missing update, checksum mismatch or synchronization failure, and retain partial evidence. No change to the matrix, precision, batch size or activation checkpointing. The process guard and allocator limit are cooperative.
