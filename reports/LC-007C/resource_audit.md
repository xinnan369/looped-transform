# LC-007C resource audit

The CPU preparation used archived metrics and generated gradients. Probe selection used the 360 original train records, with no image opens during preparation. The separately approved GPU run loaded the four existing E40 checkpoints read-only and evaluated 96 cases using the frozen train crops only. Both stages used **zero project optimizer updates and zero external experiment bytes**. Data, weights and checkpoints remain local. [Source and local checksums](probe_manifest.json)

The CPU tests and benchmark reused the installed Python environments. They did not install packages, probe mirrors, refresh weights, access official test images or access NWPU/WiderPerson. Existing synthetic optimizer smoke fixtures are part of the local test suites. New diagnostic code has no optimizer update. All commits use `[skip ci]`.

Before GPU approval, batch-two 64 px and 128 px BF16 synthetic CPU probes cost 2.51–2.72 times the generated one-backward timing. The four corresponding LC-007B real arms have epoch mean batch time p10/p50/p90 of 25.41/26.32/60.42 ms across 160 epochs. Scaling the historical p10–p90 by that CPU ratio gave a rough 6.1–15.8 s training-step cost proxy for 96 cases. Doubling the upper proxy and adding 60 s for I/O, preflight and finalization gave approximately 92 s for planning. Device dispatch, input size, backward graph and allocator differences can change these estimates. [Raw synthetic timings](gradient_equivalence.json)

## Observed approved GPU execution

| Measure | Observed | Approved limit |
| --- | ---: | ---: |
| Cases | 96 | 96 |
| Synchronized GPU-phase time | 13.414776 s | 1,200 s |
| Process peak reserved | 857,735,168 bytes (818 MiB) | 2,147,483,648 bytes |
| Project optimizer updates | 0 | 0 |
| External experiment bytes | 0 | 0 |

The 107 charged phases include .350 s startup, .929 s synthetic preflight, .752 s checkpoint loads, 11.164 s for the project cases and .220 s release. Case-phase median and maximum elapsed times were .115 and .158 s. The guard recorded no refusals, cap crossings or final synchronization errors. The four checkpoint file hashes and the exact model, EMA, optimizer and scheduler state hashes were unchanged. [Phase and state ledger](gpu_execution.json)

At launch NVML reported 19,098 MiB free and 2% utilization on the shared GPU. The same separate WSL training PID 89303 was active after completion. The available memory and observed phase times allowed the diagnostic to finish without pausing that training. The run did not compare performance with the device idle. The synchronized time excludes repository review and CPU-only archive work. [Launch and concurrency protocol](gpu_execution_protocol.md)

The complete local evidence is under `/home/xinnan/loopcrowd-lc007c-gpu`, with small system and execution logs at `/home/xinnan/loopcrowd-lc007c-gpu-background.csv` and `/home/xinnan/loopcrowd-lc007c-gpu.log`. The tracked ledger records these raw files' SHA-256 and byte counts.
