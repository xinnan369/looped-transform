# Loop-specific parameter and compute proof

The original S768 model contributes one attention, one FFN and two LayerNorms. The modified model has six additional width-192 LayerNorms and uses all eight. Each additional layer introduces 192 gamma and 192 beta values. CPU exact equality checks show all original model tensors and the post-init RNG match. The 16 affine tensors are distinct, with gamma initialized to one and beta to zero. The optimizer and EMA each cover all 63 unique trainable tensors, and EMA shadows are FP32.

| Model | Trainable parameters |
| --- | ---: |
| S768 | 1,596,614 |
| S768-LN | 1,598,918 |
| U768 | 2,931,206 |

The change adds 2,304 parameters or 0.1443% to S768. S768-LN uses 45.45% fewer parameters than U768. The option defaults to disabled and keeps the same legacy checkpoint keys and configuration digests when omitted. Resume rejects changing the enabled option.

| Forward path | S768 baseline and S768-LN dense FLOPs per image |
| --- | ---: |
| 768 training-shaped, two reads | 124,690,366,464 |
| 512 validation, four reads | 84,743,815,168 |
| 512 final-only | 41,144,025,088 |

CPU synthetic hooks measured convolution, linear and attention FLOPs. The normalization arithmetic is unchanged because all four calls already normalize twice. Initialization and inference parity on irregular padded tiles is exact for logits, offsets and the four states at the same CPU dtype and backend, with zero tolerances in FP32 and BF16. CPU tests show that all affine tensors get gradients, that the original shared affine gradient equals the sum of the separated contributions within FP32 tolerance, and that the shared attention and FFN gradients are exactly equal at initialization. [Audit data](initialization_audit.json) · [Parity tests](test_results.md)
