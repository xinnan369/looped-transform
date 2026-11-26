# Executed resources

The approved single-RTX 3090 queue completed all eight runs, 320 epochs and 57,600 successful updates in the authorized order. The GPU accounting interval charged 7,646.81 seconds, or 127 minutes 27 seconds, within the 10,800-second cap. Peak process allocated was 1,115,984,384 bytes and peak process reserved was 1,260,388,352 bytes (1.17 GiB). The process stayed within the 2 GiB limit with the frozen batch size of two. The latency probe used 72 synthetic forwards including warmup and zero optimizer updates. [Owner approval](authorization.json) · [Phase ledger](gpu_phases.jsonl) · [Execution and checkpoint evidence](run_evidence.json)

GPU snapshots recorded concurrent Python work before execution and different process IDs at final cleanup. The task made no changes to those processes. Local caches supplied all experiment inputs, with zero external experiment downloads. Hosted CI, official test, NWPU/WiderPerson and Stage 2 remain `not_run`. [Start snapshot](background_start.txt) · [End snapshot](background_end.txt)

## CPU preparation

Used local CPU environments, the verified B manifest and its cached image paths, the verified ResNet18 cache and historical K=2/K=4 reports. Tests used generated inputs, including disposable synthetic training steps. The BF16 memory proxy retained CPU storages and performed zero optimizer updates. [Input and memory audit](initialization_audit.json)
