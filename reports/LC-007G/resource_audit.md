# Executed resources

The approved single-RTX 3090 screen completed four serial runs, 160 epochs and 28,800 successful updates. The budget charged 3,609.74 seconds (60 minutes 10 seconds) against the 5,400-second ceiling, including 20.38 seconds for CPU preflight. Process peak allocated was 775,961,600 bytes and peak reserved was 983,564,288 bytes (0.92 GiB). The runner observed memory without applying a process quota or allocator fraction. The rotated latency probe performed 24 synthetic forwards including warmup and zero optimizer updates. [Authorization](authorization.json) · [Phase ledger](gpu_phases.jsonl) · [Run evidence](run_evidence.json)

The existing Python process with PID 139343 appeared in the start and end snapshots. This run left it in place. Available device memory was sampled at the start and end. Existing local caches supplied all experiment inputs. External experiment downloads were zero. Hosted CI, official test, NWPU/WiderPerson and Stage 2 remain `not_run`. [Background start](background_start.txt) · [Background end](background_end.txt) · [Device memory start](device_memory_start.json) · [Device memory end](device_memory_end.json)

## CPU preparation

Used existing local Windows and WSL environments with CUDA hidden, generated test tensors and images, cached B metadata and cached ResNet18 weights. The CPU audit verifies historical CSVs, full-state/RNG parity and matched crop/LR trajectories. The reachability probe uses model-generated synthetic tokens only. [Input audit](initialization_audit.json)
