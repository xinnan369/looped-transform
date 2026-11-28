# LC-007K resource audit

The one WSL RTX 3090 attempt ran on 2026-11-28 from 10:10:56 to 10:32:21 Asia/Shanghai, with exit code 0. Execution commit `06205f38d847f725e9a621aa276d0a0d11186d90` and config SHA-256 `7d78b00ba75eb87cc6165d2945ecbd1422fd7f338056db52ec388ed5412f1415` match the scoped launch instruction. The authorization records the absence of independent review at launch. [Authorization](authorization.json) · [Execution evidence](run_evidence.json)

| Measure | Observed |
| --- | ---: |
| Launcher wall time | 21 min 25 s |
| Charged time / ceiling | 1,283.194 / 1,800 s |
| Runs / epochs / successful updates | 1 / 40 / 7,200 |
| Peak process allocated / reserved | 1,487,092,736 / 1,767,899,136 bytes |
| Peak process reserved | 1,686 MiB |
| `nvidia-smi` free before launch | 18,539 MiB |
| CUDA runtime free at start / end | 24,436,015,104 / 22,567,452,672 bytes |
| Local free disk before launch | 95,501,889,536 bytes |
| External downloads / disposable updates | 0 / 0 |

The device reads came from separate interfaces and sampling times. Background compute PID 246664 appears in both preflight and cleanup snapshots. The 164 charged phases have zero failures. No quota, OOM, timeout or retry occurred.
