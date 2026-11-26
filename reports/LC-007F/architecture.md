# Eight-call factorial

The four arms combine shared or independent fixed-window blocks with ungated or channel-gated recurrence. Each step uses width 192, an 8-pixel token window, six heads, the common ResNet18 projection and the existing point decoder. Gated models apply unit-initialized deltas to the attention and FFN residuals and the full post-residual state, using the LC-007D formulas. Their FP32 deltas are registered after block construction without advancing the RNG. Broadcast NHWC scales cast to the branch or state dtype. [Gate equations](../LC-007D/architecture.md)

| Arm | Independent blocks | Extra gate scalars | Parameters |
| --- | ---: | ---: | ---: |
| S8 | 1 | 0 | 1,596,614 |
| U8 | 8 | 0 | 4,710,662 |
| S8RS | 1 | 4,608 | 1,601,222 |
| U8RS | 8 | 4,608 | 4,715,270 |

Per native 512 tile the common dense FLOP estimates are 55,696,171,008 for final-only, 70,229,434,368 for midpoint/final and 157,429,014,528 for all eight readouts. Gates add 4,608 scale additions and 18,874,368 multiplications per forward. The estimator uses two operations per MAC and excludes feature-norm reductions, dtype conversions and transfer costs. [Measured CPU FLOP accounting](initialization_audit.json)

The deployment call decodes only at depth eight. Training uses the explicit midpoint/final path and two losses. Validation selects the fixed readout schedule and never makes a second forward for additional stages. The existing K=2/4 checkpoint keys and model paths are preserved. GPU latency and process memory are `not_run` for K=8. [Schedule](protocol.md)
