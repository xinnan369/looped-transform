# LC-007I training resource audit

The single frozen attempt ran from 08:08:07 to 08:27:46 Asia/Shanghai on 2026-11-28 and exited with code 0. Execution used commit `8dd00d0b9f78cc954a92824f868825ed786f4fd1`, tree `4956ad20fc5e667eb4d0627649be5b61389724d0` and execution config SHA-256 `f14c9befb3f7a434f82f723a54831f00eef3c21d8b4be0789dd9c7840aed80e8`. The operator recorded the owner's direct launch instruction after the implementer's technical recheck. [Authorization summary](authorization.json) · [Phase and file evidence](run_evidence.json)

| Measure | Observed |
| --- | ---: |
| Launcher wall time | 19 min 39 s |
| Charged time / ceiling | 1,174.758 / 1,800 s |
| Runs / epochs / optimizer updates | 1 / 40 / 7,200 |
| Peak process allocated | 1,463,051,264 bytes, 1,395.27 MiB |
| Peak process reserved | 1,736,441,856 bytes, 1,656 MiB |
| RTX 3090 `nvidia-smi` free just before launch | 18,835 MiB |
| CUDA runtime free at preflight / cleanup | 24,436,015,104 / 22,598,909,952 bytes |
| Local disk free before launch | 95,667,073,024 bytes |
| External downloads / disposable updates | 0 / 0 |

The two GPU memory interfaces were sampled at different times. Background compute PID 228689 appeared at both preflight and cleanup. The 164 phase records contain zero failures. The launcher used no allocator quota, with zero OOM, timeout or retry events.
