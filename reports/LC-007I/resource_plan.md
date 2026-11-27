# Single-run resource proposal

One RTX 3090, one run, 40 epochs and 7,200 successful updates. Proposed ceiling is 1,800 charged seconds, with zero new downloads and zero disposable GPU updates. The launcher uses TERM at 1,795 seconds and a five-second hard-kill allowance. Wall accounting covers CPU preflight, GPU setup, training, validation, transfers, checkpoints, logging, intervals, failures and cleanup.

| Basis | Seconds |
| --- | ---: |
| Historical training | 281.13 |
| Historical live validation | 295.98 |
| Historical EMA validation | 292.18 |
| Historical initialization and checkpoints | 1.99 |
| Phase-specific projection using 2.25× only for training | 1,222.69 |
| 25% allowance plus 120 s | 425.67 |
| Total CPU-based plan | 1,648.36 |

The projection uses the larger of area growth and measured CPU dense FLOP growth for the historical training phase. Validation remains fixed. Runtime rechecks that this estimate fits the ceiling, local disk has at least 3 GiB free, and measured device headroom covers the 4,265,607,168-byte historical peak-based estimate. This is a launch availability check. The process applies no fixed/dynamic VRAM quota or allocator fraction. Record actual free memory, process peaks, phase times and concurrent process observations without disturbing other jobs.

The CPU preparation snapshot reported 97,514,692,608 bytes free on WSL disk and 18,787 MiB free on the RTX 3090. A separate compute process was present and left running. The snapshot is informational and the runner rechecks physical availability at launch.

Keep the failed attempt's logs and checkpoints if real OOM, time, nonfinite, cache, trajectory or parity errors occur. The frozen queue does not retry or alter batch, precision or data. No official test or external data access is scheduled. Independent review of the exact freeze is pending. [Freeze and limits](protocol.md)
