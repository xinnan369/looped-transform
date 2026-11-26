# CPU verification

The suites ran from freeze `2dc472c602eeb09c646e1085547a8d716e1136ab`, on existing environments with CUDA hidden. WSL checked out a local Git bundle on its native filesystem.

| Environment | Full result | Local log | SHA-256 |
| --- | --- | --- | --- |
| WSL Python 3.10.16, `mamba-ssm` | 142 passed, 35.59 s | `artifacts/lc007e-py310.log` | `7fe1d6fde4cbdd2062612825a2302a8ebea38c5d084abe3858eedaf00ae2b7bd` |
| WSL Python 3.11.11, `sigma-r1-cu121` | 142 passed, 34.80 s | `artifacts/lc007e-py311.log` | `0454b7db6e40555c8f2bc2c1582bd0513a97255bc53790b8be3ae330414afad6` |
| Windows Python 3.13.2 | 142 passed, 61.90 s, seven deprecation warnings | `artifacts/lc007e-py313.log` | `149c4ba404fcf6052b515b51e957e3aa3aaa70b50ddaea692bc70810a03cab07` |

The new tests verify FP32/BF16 identity-gate outputs and offsets, both supervised losses and all existing gradients, four active state-gradient paths, midpoint isolation from later gates, and tied/untied parameter object identity. Tiled tests verify four reads in one forward, with the final predictions matching the deployment path. The synthetic EMA fit checks per-policy stage norms, one shadow per unique parameter and checkpoint restore. Budget refusal and trajectory mismatch tests verify stop behavior and retained interruption evidence. [Tests](../../tests/test_depth4.py)

The separate audit checked 24 LC-007B and 18 LC-007D indexed files, all eight planned starts and 320 planned epoch crop/LR pairs against archived actual pairs. It checked the seed-matched encoder, decoder and inherited block digests, unchanged RNG for gated/ungated pairs, and all four block digests for the matched unshared comparison. The audit digest is `7bf48295ffa2dc15948281a0469105130d0130bea15eee5761a9239a30fba8b0`. [Audit](initialization_audit.json)

Post-freeze archival fixtures exercise the joint criteria, sharing interactions and partial queue path. The partial fixture read the four completed seed-42 histories, verified their checkpoint and trajectory digests and checked 56,480 archived rows and 1,920 gate rows. [Archiver](../../scripts/archive_lc007e.py)

Synthetic parity, budget checks and arithmetic estimates characterize the CPU preparation. Hosted CI and historical checkpoint loading remain `not_run`. [Executed resources](resource_audit.md)
