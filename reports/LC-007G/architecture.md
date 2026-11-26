# Fixed-depth shift geometry

S4-shift and U4-shift use the existing four-call fixed-window blocks with origin offsets 0, 4, 0, 4, width 192, 8×8 token windows, six heads and stride-8 P3. Padding plus a key mask implements a noncyclic origin shift. Window reversal crops padding before residual and FFN paths. S4-shift uses the same Parameter objects on all calls. U4-shift uses four independent initialized blocks. Both reuse the backbone and point decoder. No trainable parameters are added. S4 and S4-shift have 1,596,614 parameters. U4 and U4-shift have 2,931,206.

The existing `shifted_shared_4` final-only path and checkpoint keys retain their behavior. Opt in via `depth_screen` to enable midpoint/final training and four readouts. Each training pass computes only the losses at calls 2 and 4. The default forward uses the final-only decoder. [Model](../../src/loopcrowd/model/__init__.py) · [Validated config](../../configs/lc007g_cross_window.json)

| Native 512 tile, dense FLOPs | Fixed window | Shifted | Added |
| --- | ---: | ---: | ---: |
| Final only | 40,395,341,824 | 41,144,025,088 | 748,683,264 |
| Midpoint and final | 54,928,605,184 | 55,677,288,448 | 748,683,264 |
| Four readouts | 83,995,131,904 | 84,743,815,168 | 748,683,264 |

On the 64×64 P3 grid, calls 1 and 3 use 64 windows. Calls 2 and 4 pad to 72×72 and use 81 windows. The estimate counts two operations per MAC, with attention and its projections on padded tokens, FFN on the original grid and the decoder on each requested read. Shifted final-only compute grows about 1.85%. The matched shifted arms use the same arithmetic. The estimate excludes state diagnostics, transfers and kernel overhead. [CPU hook evidence](initialization_audit.json)

The charged RTX 3090 BF16 preflight produced final-only medians of 7.445 ms (S4-shift) and 9.936 ms (U4-shift), and four-read medians of 8.630 ms and 7.204 ms. Each median uses five synchronized 512-pixel synthetic batch-one samples with rotated arm order after warmup. Concurrent work was present, and the sample ordering and variance limit throughput interpretation. [Latency samples](synthetic_gpu_latency.json)
