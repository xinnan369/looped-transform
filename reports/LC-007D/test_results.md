# LC-007D local CPU verification

Each full suite ran from the frozen code and protocol at `6f6c37a2c3643e1b97ec0d1f27d53cdf2b60b695` in an existing environment with CUDA hidden. The WSL runs used a local Git bundle and a WSL-native checkout.

| Existing environment | Full result | Local log | SHA-256 |
| --- | --- | --- | --- |
| Ubuntu 22.04 WSL, Python 3.10.16, `mamba-ssm` | 131 passed in 29.04 s | `artifacts/lc007d-py310.log` | `9fd14fc46755e6fa95743b0b0ae55665990ac95d440373422aa12ae87fbf7b32` |
| Ubuntu 22.04 WSL, Python 3.11.11, `sigma-r1-cu121` | 131 passed in 27.94 s | `artifacts/lc007d-py311.log` | `f641281a43668c518febe0c5a44e22ef80575d52fbc5d3e6d1880cc4e15920a1` |
| Windows, Python 3.13.2 | 131 passed in 52.06 s, six dependency/API deprecation warnings | `artifacts/lc007d-py313.log` | `b2ff9f086a89b94a93c9ba0ebef391c398b9dd3a3786aa23fb6fa7a73c211070` |

The new tests compare exact encoder, decoder, block and post-construction RNG hashes, test zero-delta equivalence in FP32 and BF16 for stage outputs, losses, existing parameter gradients and gate gradients, and check absence of stage-two gradients from an early-only loss. Nonuniform scales verify each residual, the complete state update, trace names, stage separation and post-update decode. Tests also exercise per-branch FLOP accounting, EMA shadow names, loading a gated checkpoint, reporting both weight policies in `fit`, stopping on trajectory mismatch and rejection of missing or mismatched approval records. The full suites include the legacy scalar-gain, stage readout, checkpoint and LC-007C diagnosis regressions. [New test cases](../../tests/test_channel_gates.py) · [Frozen tolerances](../../configs/lc007d_channel_screen.json)

The separate CPU input audit reconciled 24 indexed files, eight historical runs with all 40 crop/LR hashes each, and six new starting states with the local cache. Five batch-one final-only synthetic CPU forwards per model yield the median rates below, in milliseconds. [Raw audit and timings](initialization_audit.json)

| Ungated Shared-2 | R | RS | Ungated Unshared-2 | U-RS |
| ---: | ---: | ---: | ---: | ---: |
| 191.52 | 193.09 | 192.96 | 190.35 | 190.09 |

Project updates, GPU timing and peak memory, and hosted CI are `not_run`. The local metrics here describe CPU verification and synthetic inputs. [Resource audit](resource_audit.md)
