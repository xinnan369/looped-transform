# Training resource audit

The frozen queue ran on one WSL RTX 3090 from 12:45:01 to 14:20:18 Asia/Shanghai on 2026-11-27 and exited with code 0. It ran commit `4696be2fca77fcd534260dc0d8bc07f38d098e5a`, source tree `fa412ab0945d13a6cd0744e3bcd37c460120c95e` and configuration SHA-256 `12499b6c3246fccd6c7c4342d6237a0c15e9cdd15f112390224716e9b5d2e5b9`. [Authorization](authorization.json) · [Execution and phase ledger](run_evidence.json)

| Measure | Observed |
| --- | ---: |
| Launcher wall time | 95 min 17 s |
| Charged time / authorized ceiling | 5,715.447 / 7,200 s |
| Completed runs / epochs / successful updates | 6 / 240 / 43,200 |
| Peak process allocated / reserved | 779,942,912 / 979,369,984 bytes |
| Peak process reserved | 934 MiB |
| `nvidia-smi` free memory immediately before launch | 19,210 MiB |
| CUDA runtime free at preflight / cleanup | 24,436,015,104 / 23,355,981,824 bytes |
| Local disk free immediately before launch | 96,628,297,728 bytes |
| External download bytes / disposable GPU updates | 0 / 0 |

The runtime and `nvidia-smi` values were recorded by separate interfaces at different times. The background compute PID 181319 appeared in both the start and end snapshots. The queue used no allocator quota and recorded no OOM, timeout, phase failure or retry. Resource evidence retains hashes of the observed snapshots and raw runner log. [Runtime provenance](run_evidence.json)
