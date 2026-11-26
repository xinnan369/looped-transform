# Four-call factorial

S4 and U4 use existing fixed-window K=4 backbones with one reused block or four independently initialized blocks. S4RS and U4RS apply the same three unit-initialized channel scales as LC-007D on the attention residual, the FFN residual and the full post-residual state. Each gated arm trains 4 × 3 × 192 = 2,304 FP32 deltas, initialized after the blocks without advancing the RNG. NHWC scales cast to branch or state dtype for multiplication. The common width is 192, the window is 8 and attention has six heads. The same decoder reads all stages. [K=2 gate definition](../LC-007D/architecture.md)

| Arm | Transformer block objects | Additional gate parameters | Total parameters |
| --- | ---: | ---: | ---: |
| S4 | 1 | 0 | 1,596,614 |
| U4 | 4 | 0 | 2,931,206 |
| S4RS | 1 | 2,304 | 1,598,918 |
| U4RS | 4 | 2,304 | 2,933,510 |

The dense estimate per 512 tile is 40,395,341,824 operations with final-only decode and 83,995,131,904 for four reads. Two additional validation reads relative to the training readout add 29,066,526,720 dense operations. Per gated tile add 2,304 scale additions and 9,437,184 multiplications. The estimator uses two operations per MAC for Conv/Linear/MHA and excludes state-norm bookkeeping and memory traffic. The historical K=2 two-read recipe costs 47,278,194,688 dense operations. [Initialization and cost evidence](initialization_audit.json)

The explicitly selected depth training path decodes twice and uses one auxiliary loss at call 2. Four diagnostic reads share the same tiled forward. The ordinary final-only API and the legacy two-call API retain their established behavior. [Model](../../src/loopcrowd/model/__init__.py) · [Tiled readout](../../src/loopcrowd/inference.py)

| Arm | BF16 final-only median ms | BF16 four-read median ms |
| --- | ---: | ---: |
| S4 | 5.842 | 7.167 |
| U4 | 6.037 | 7.281 |
| S4RS | 6.678 | 8.131 |
| U4RS | 6.167 | 7.957 |

Each result uses five synchronized synthetic 512-pixel forward samples with rotating order after warmup. Other GPU work was present. The latency probe measures decoder forward cost without validation tiling, state-norm bookkeeping or CPU matching. [Raw latency samples](synthetic_gpu_latency.json)
