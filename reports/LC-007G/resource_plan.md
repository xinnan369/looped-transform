# Proposed shifted-window GPU scope

**GPU_TRAINING_NOT_AUTHORIZED.** Propose one serial RTX 3090, four fresh runs, 160 epochs, 28,800 successful updates, at most 5,400 charged seconds, zero external experiment downloads and zero disposable GPU optimizer updates. The resource policy observes allocated and reserved peaks and available device memory. There is no fixed process VRAM budget and no allocator fraction for LC-007G. Prior experiment approvals and resource records retain their original scopes. [Issue #29 policy](https://github.com/xinnan369/looped-transform/issues/29) · [Frozen approval schema](../../configs/lc007g_cross_window.json)

| Phase | Projected seconds |
| --- | ---: |
| Training | 1,186 |
| Live validation | 1,227 |
| EMA validation | 1,216 |
| Checkpoint | 10 |
| Preflight, transfer, interphase and cleanup reserve | 120 |
| Central estimate | 3,759 |
| With 20% contingency | 4,511 |

The eight-run LC-007E ledger charged 7,385.73 seconds. Half is about 3,693 seconds, while the projection uses the actual four matched fixed S4/U4 phase sums. Training and validation scale respectively by the 1.01363 midpoint/final and 1.00891 four-read FLOP ratios. With contingency the estimate is about 75.2 minutes, leaving roughly 889 seconds before the proposed ceiling. GPU kernels, CPU matching and contention can move actual time. [Historical device ledger](../LC-007E/gpu_phases.jsonl) · [Projection inputs](initialization_audit.json)

The headroom check compares the current free device memory at launch with a working-set estimate of 1,296,007,168 bytes. It scales the observed prior K=4 peak by 81/64 for padded windows and adds 64 MiB. This estimates available physical headroom before launch. The process may reserve more when device memory is available, and the guard records actual peaks without rejecting by a per-process threshold. Check concurrent jobs and preserve their execution. The runner needs 3 GiB free local disk for retained checkpoints, interruption evidence and metrics.

The deferred runner requires dated owner approval linked to the original message and exact commit, plan digest and resource limits. It performs CPU input verification before initialize-device work. The charged synthetic BF16 preflight rotates arm order and alternates final-only and four-read calls, with one warmup and five timing repeats per arm/mode. Log observed contention and device memory before and after. The photo count source remains unchanged and latency is measured using synthetic tiles. [Deferred runner](../../scripts/run_lc007g.py)

Charge setup, synchronization, preflight, training, validation, transfers, serialization, interphase work, cleanup and failures. The conservative floor is 30 seconds for preflight, 20 for each train/validation phase, two for checkpoint and 20 for cleanup, with observed phase maxima for later checks. Stop on projected time overrun, demonstrably insufficient launch headroom, OOM, nonfinite, missing data, update or hash mismatch, and synchronization failure. Retain partial evidence and the frozen batch size, precision and optimizer. [Executed resources](resource_audit.md)
