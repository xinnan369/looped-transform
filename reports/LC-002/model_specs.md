# Fixed-loop model and profiling specification

Every variant uses torchvision ResNet-18 through layer2. Stem + layer1 provide 64-channel P2 at stride 4. Layer2 provides 128-channel P3 at stride 8 and a 1 × 1 projection makes 192-channel tokens. P3 is resized to the exact P2 spatial shape, concatenated with P2 and passed through a 3 × 3 convolution, GroupNorm and ReLU. Identical 1 × 1 heads produce confidence and offsets.

The Transformer block is pre-norm residual attention plus pre-norm residual 192 → 768 → 192 GELU FFN. Attention uses non-overlapping 8 × 8 windows, six 32-channel heads and a key-padding mask. Unpartitioning restores the original token shape. Shared variants repeatedly call one block object. The four-block control registers four independent blocks of exactly the same architecture. CNN-only still uses the same P3 projection and fusion. Optional loop embedding, density and gradient checkpointing are disabled for the measured controls. Each initialized experiment uses the same seed before building all common modules.

For input `[B,3,H,W]`, confidence is `[B,K,ceil(H/4),ceil(W/4)]`. Offsets are `[B,K,2,ceil(H/4),ceil(W/4)]`. Optional density is `[B,1,ceil(H/4),ceil(W/4)]`. At default K=2 and 512², the spatial grid is 128 × 128. Complete inference and target policy are in the [data and evaluation protocol](../../docs/DATA_AND_EVAL_PROTOCOL.md).

| Variant | Calls | Distinct blocks | Parameters | Forward GFLOPs / 512² image |
|---|---:|---:|---:|---:|
| cnn_only | 0 | 0 | 1,151,750 | 25.095 |
| shared_1 | 1 | 1 | 1,596,614 | 28.920 |
| shared_2 | 2 | 1 | 1,596,614 | 32.745 |
| shared_4 | 4 | 1 | 1,596,614 | 40.395 |
| unshared_4 | 4 | 4 | 2,931,206 | 40.395 |

FLOPs count two per multiply-accumulate for convolutions, linear layers, Q/K/V and output projections, and the two attention matrix products. Hooks run on every invocation, including repeated use of the shared block. The estimate excludes normalization, activations, bias adds, softmax, upsampling and shape operations. Latency uses five synchronized measurements after warm-up. The train step includes loss, backward and AdamW on resident, synthetic batch 2 inputs. The eval runs measure batch-one forward on a real validation tile, and separately batch-one tiled prediction and stitching on its full 1021 × 611 frame. CUDA peak allocated/reserved numbers refer to the train step.

Torchvision 0.17.2 supplies the ResNet implementation under BSD-3-Clause. The pilot config requests its `ResNet18_Weights.IMAGENET1K_V1` checkpoint, whose URL, file hash and actual load status are recorded in each run manifest. The profile and CPU fixtures use random initialization. Weights remain external. [Torchvision source license](https://github.com/pytorch/vision/blob/v0.17.2/LICENSE) · [Checkpoint documentation](https://docs.pytorch.org/vision/0.17/models/generated/torchvision.models.resnet18.html)
