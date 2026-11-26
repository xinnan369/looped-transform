# Proposed K=4 resource envelope

**GPU_TRAINING_NOT_AUTHORIZED.** Proposed scope is one visible RTX 3090, eight serial fresh runs, 320 epochs, 57,600 successful project updates, at most 9,000 charged seconds, at most 2 GiB peak process reserved, zero disposable optimizer updates and zero external experiment bytes.

| Phase | Projected seconds |
| --- | ---: |
| Training | 2,217 |
| Live validation | 2,588 |
| EMA validation | 2,539 |
| Checkpoint serialization | 30 |
| Preflight, setup, transfers and cleanup reserve | 90 |
| Central estimate | 7,464 |
| With 15% contingency | 8,584 |

LC-007D measured 3,619 seconds for six K=2 runs. Scaling to eight K=2 runs gives about 4,825 seconds. The K=4 projection scales observed D phase averages to 320 epochs with dense compute ratios 1.162 for training and 1.777 for each validation policy, doubles historical checkpoint cost and reserves 90 seconds. The 15% increment covers uncertainty in backward scaling, CPU matching, state-norm reductions, transfers and contention. GPU kernels, data-dependent matching and concurrent work may depart from these ratios. Current buffer to the proposed ceiling is 416 seconds with that contingency. [Reproducible projection](initialization_audit.json)

LC-007D reserved at most 824,180,736 bytes. A twofold scenario gives 1,648,361,472 bytes (1.54 GiB). This is a feasibility scenario. Four-call activation retention, unshared optimizer state and GPU allocator behavior require process observation after approval. The largest model has fewer than three million parameters. At 16 bytes per parameter for FP32 weights, AdamW moments and EMA, the eight retained final checkpoints span under 384 MB before serialization metadata. The runner requires 3 GiB free for those checkpoints, an interrupted checkpoint, per-image history and evidence.

CPU preflight verifies cache hashes and paired schedules before initializing CUDA. After separate authorization the runner verifies single compatible RTX 3090, captures concurrent work, requires at least 2 GiB free, and sets a 2 GiB allocator fraction. Existing jobs continue. The synthetic latency probe uses generated 512-pixel BF16 batch-one inputs and rotates four arm positions and final-only vs four-read ordering with one warmup and five synchronized samples per arm and mode. Record contention from start/end snapshots. The probe measures isolated synthetic calls within this process, and performs zero optimizer updates. Host/GPU contention is allowed.

Synchronized phases, transfers, interphase work, serialization, failures and cleanup are charged from the GPU preflight through final cleanup. Keep a 15-second cleanup reserve and conservatively budget the next phase against historical defaults and observed maxima. Check at training batches and validation images. Stop if the CPU projection, upcoming phase, memory, nonfinite check, observed steps/crop/LR hashes, synchronization or cleanup fails. Keep completed and failed evidence. A resource stop does not broaden the scope. [Deferred runner](../../scripts/run_lc007e.py)

The runner requires a dated record of the owner's explicit approval for Issue #25, reviewed commit, screen-config digest and exact limits. The record must refer to the original message. The freeze contains no such approval. [Issue #25 scope](https://github.com/xinnan369/looped-transform/issues/25)
