# Executed resources

The approved RTX 3090 screen completed eight serial runs with 320 epochs and 57,600 successful updates. It charged 7,385.73 seconds (123 minutes 6 seconds) against the 9,000-second ceiling. Process peak allocated reached 804,309,504 bytes and peak reserved reached 970,981,376 bytes (0.90 GiB). Preflight recorded five synchronized samples plus warmup for each of four arms and two readout modes, for 48 synthetic forwards and zero optimizer updates. [Owner approval](authorization.json) · [Phase ledger](gpu_phases.jsonl) · [Execution and checkpoint evidence](run_evidence.json)

The start snapshot showed an existing Python compute process. The end snapshot showed Python processes with different PIDs, including this runner during final cleanup. No concurrent process was paused, stopped or reconfigured by this task. The approved run reused existing caches and downloaded zero external experiment bytes. Hosted CI, official test, NWPU/WiderPerson and Stage 2 remain `not_run`. [Start snapshot](background_start.txt) · [End snapshot](background_end.txt)

## CPU preparation

The CPU scope used the existing Windows and WSL Python environments. It checked local B manifest metadata, cached image paths, cached ResNet18 SHA-256 and historical CSVs. Tests used synthetic examples. CPU FLOP counts came from synthetic forwards. [Input audit](initialization_audit.json)
