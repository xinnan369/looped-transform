# Eight-call factorial

The four arms combine shared or independent fixed-window blocks with ungated or channel-gated recurrence. Each step uses width 192, 8×8 token windows, six heads, the common ResNet18 projection and the existing point decoder. Gated models apply unit-initialized deltas to the attention and FFN residuals and the full post-residual state, using the LC-007D formulas. Their FP32 deltas are registered after block construction without advancing the RNG. Broadcast NHWC scales cast to the branch or state dtype. [Gate equations](../LC-007D/architecture.md)

| Arm | Independent blocks | Extra gate scalars | Parameters |
| --- | ---: | ---: | ---: |
| S8 | 1 | 0 | 1,596,614 |
| U8 | 8 | 0 | 4,710,662 |
| S8RS | 1 | 4,608 | 1,601,222 |
| U8RS | 8 | 4,608 | 4,715,270 |

Per native 512 tile the common dense FLOP estimates are 55,696,162,816 for final-only, 70,229,426,176 for midpoint/final and 157,429,006,336 for all eight readouts. Gates add 4,608 scale additions and 18,874,368 multiplications per forward. The estimator uses two operations per MAC and excludes feature-norm reductions, dtype conversions and transfer costs. [Measured CPU FLOP accounting](initialization_audit.json)

The deployment call decodes only at depth eight. Training uses the explicit midpoint/final path and two losses. Validation selects the fixed readout schedule and never makes a second forward for additional stages. The existing K=2/4 checkpoint keys and model paths are preserved. [Schedule](protocol.md)

| Arm | Final median ms | Midpoint/final median ms | Eight-read median ms |
| --- | ---: | ---: | ---: |
| S8 | 9.293 | 11.029 | 11.175 |
| U8 | 13.914 | 17.465 | 30.475 |
| S8RS | 12.493 | 16.544 | 30.699 |
| U8RS | 15.155 | 17.140 | 19.520 |

The RTX 3090 preflight used five synchronized BF16 batch-one synthetic 512-pixel samples after warmup in each mode and rotated arm order. These timings exclude tiled validation, feature-norm reductions and CPU matching. Concurrent work was present and timings varied across samples. [Latency samples](synthetic_gpu_latency.json) · [Resource audit](resource_audit.md)
