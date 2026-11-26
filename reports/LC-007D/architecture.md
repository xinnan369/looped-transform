# Fixed-depth channel gates

Burzio et al. use unit-initialized scales on the attention and MLP residuals and on the state of a looped Transformer. LC-007D adapts the channel scaling in §3.4 and Table 4 to two fixed learned stage vectors per branch for head-point counting. This fixed-depth discretization preserves the width-192 pre-norm window block and shared decoder. It introduces no continuous-time MLP, additional embeddings, shifted windows, LayerScale, head, backbone or loss. [Déjà View, §3.4 and Table 4](https://arxiv.org/html/2605.30215v1)

For stage \(t\), form \(x' = x + (1+a_t)\odot \mathrm{Attn}(\mathrm{LN}_1(x))\) and \(u = x' + (1+b_t)\odot\mathrm{FFN}(\mathrm{LN}_2(x'))\). R returns \(u\). RS and U-RS return \((1+c_t)\odot u\), scaling the complete residual state before the corresponding point decoder. R and RS call one shared block twice. U-RS makes one call to each independent block. The early output is decoded after the first stage's state update. The existing scalar-sigmoid `gain_logits` and legacy block trace retain their prior semantics.

The gate deltas are a zero-initialized FP32 parameter added after common modules and blocks without advancing the random stream. Layout is `[two stages, branches, 192]`. Scales are computed in FP32, then cast to the attention, FFN or state tensor dtype at each NHWC multiplication. FP32 gate gradients flow through the cast. The new `channel_trace` exposes the state entering and leaving each stage and its named FP32 scales.

| Arm | Parameters | Extra parameters | Gate scale adds per 512 tile | Gate multiplications per 512 tile |
| --- | ---: | ---: | ---: | ---: |
| Ungated Shared-2 | 1,596,614 | 0 | 0 | 0 |
| R | 1,597,382 | 768 | 768 | 3,145,728 |
| RS | 1,597,766 | 1,152 | 1,152 | 4,718,592 |
| Ungated Unshared-2 | 2,041,478 | 0 | 0 | 0 |
| U-RS | 2,042,630 | 1,152 | 1,152 | 4,718,592 |

The common dense estimate is 32,744,931,328 operations per 512 tile for final-only outputs under the repository's convolution/linear/MHA convention of two operations per MAC. Gates add the tabled arithmetic at stride eight with a 64×64 token grid. The estimator excludes casts, additions already present in the old model and memory movement. CPU measurements use five synchronized-by-return FP32 batch-one synthetic forwards after warmup. [Raw profiles and initialization digests](initialization_audit.json)

LC-004A tested stage and tied scalar residual gains and did not pass its gate. LC-007C found zero negative cosine across 240 defined shared-parameter directions and smaller second-call local nAP gain than the matched unshared reference. Channel gating tests a separate architecture hypothesis. Whether it improves early-to-final correction remains unevaluated. [Scalar gain results](https://github.com/xinnan369/looped-transform/pull/10) · [Stagewise diagnosis](https://github.com/xinnan369/looped-transform/pull/22)
