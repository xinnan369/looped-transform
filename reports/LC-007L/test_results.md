# CPU verification

Execution freeze `17eb61fda1e94b73fca9bc539f0c4499622950da`. All three test processes hid CUDA. The two WSL suites ran from a clean checkout of the local bundle, and shell launcher syntax passed.

| Environment | Full suite | Local log SHA-256 |
| --- | --- | --- |
| WSL Python 3.10.16 | 215 passed in 78.22 s | `9853144901c6dba8f42d5f67b7f3cf559407b69b251202a6c5224c2b6af54da2` |
| WSL Python 3.11.11 | 215 passed in 80.02 s | `e5753d1ac2ddb76206ac52f24fd781c2d6337293d7c3b7598e735d01974ad296` |
| Windows Python 3.13.2 | 215 passed in 120.88 s, 13 deprecation warnings | `29a6f3d4aced36375d3c1729bfd23cf9b24f545b296db120a49bab931e39263f` |

Seven added cases cover exact FP32/BF16 initialized four-state and logits/offsets parity, final-only/two/four-read agreement, fixed and shifted irregular windows, unique affine gradients and divergent updates, shared gradient aggregation, strict checkpoint and resume behavior, optimizer/EMA participation with live BN, and archived diagnostic shape, restricted configuration, authorization and joint scoring. The inherited suites cover legacy model/checkpoint and config digest parity, data, masking and resource guards.

The clean CPU preflight reproduced a SHA-256 of `d97d6b2f91cee932235b1ca17da9cb4586c90ae942dee45d67aec72f7937259d`. It verified both native 768 exposure histories, all 400 cached image geometries and matched historical initialization, including the exact state of the one retained shared block. No GPU optimization or disposable GPU workload occurred. [Preflight](initialization_audit.json)
