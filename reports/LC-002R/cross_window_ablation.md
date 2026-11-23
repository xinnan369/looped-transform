# Alternating recurrent windows

The shifted variants alternate origins on passes 0/1/2/3 as 0/4/0/4 tokens along both axes. They retain the same 192-wide, six-head WindowBlock object for all calls. Original unshifted variants always use origin 0. Padding produces shifted windows without wrap-around. The ordinary key-padding mask marks every synthetic position and window reversal slices to the original P3 shape.

A block-only regression perturbs the token to the right of the original x=8 window boundary before call two. The adjacent output across the old boundary changes with shift 4. With origin 0 the same output is unchanged. Perturbing both opposite edges leaves the near-corner output unchanged. The shifted tests also check padded positions, backward and Parameter identity.

For 512 × 512 input, P3 is 64 × 64. A fixed pass evaluates 64 windows. A shifted pass evaluates 81 windows over a padded 72 × 72 layout. The extra window products contribute to FLOPs and memory. The training config, code and external artifact hashes are in [training history](training_history.json).

| Variant | Parameters | GFLOPs/image | Peak allocated MiB | Peak reserved MiB | Batch-2 step ms | Batch-1 tile ms | Full frame ms | Val MAE at 0.30 | Local nAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_2 | 1,596,614 | 32.745 | 496.0 | 666.0 | 14.63 | 4.53 | 61.22 | 181.90 | 0.1793 |
| shifted_shared_2 | 1,596,614 | 33.119 | 502.9 | 550.0 | 16.59 | 4.28 | 57.60 | 176.27 | 0.1873 |
| shared_4 | 1,596,614 | 40.395 | 610.5 | 654.0 | 19.77 | 5.38 | 72.60 | 192.50 | 0.1816 |
| shifted_shared_4 | 1,596,614 | 41.144 | 622.7 | 798.0 | 20.28 | 7.88 | 81.95 | 208.73 | 0.1757 |
| unshared_4 | 2,931,206 | 40.395 | 628.3 | 680.0 | 22.14 | 5.17 | 75.89 | 171.23 | 0.1896 |

FLOPs count two operations per multiply-accumulate for conv/linear/QK/AV and projections. Timings are medians of nine synchronized repeats after warm-up on one RTX 3090. Full-frame timings include decode and ownership assignment on a 1021 × 611 validation image. Localization is the local diagnostic on in-frame candidates above the score floor.

At seed 42, the shifted two-call control had a lower best fixed-cutoff MAE and higher local nAP than its unshifted counterpart. The shifted four-call control had higher MAE and lower local nAP than its unshifted counterpart. Both shifted controls exceeded 25% late validation MAE CV. The block test establishes cross-window communication, and the measured validation outcomes vary with loop depth.

## Seam ambiguity at the selected checkpoints

| Variant | Above-threshold candidates outside their source ownership | Owned cross-tile pairs within 2 px | Reconciliations |
|---|---:|---:|---:|
| cnn_only | 16222 | 9 | 0 |
| shared_1 | 15670 | 10 | 0 |
| shared_2 | 15460 | 8 | 0 |
| shared_4 | 14770 | 12 | 0 |
| unshared_4 | 14448 | 11 | 0 |
| shifted_shared_2 | 15390 | 7 | 0 |
| shifted_shared_4 | 11992 | 7 | 0 |

Ownership exclusions count predictions outside the contributing tile's owner rectangle. Their association to distinct ground-truth heads is unknown. An owned cross-tile pair can represent a shifted duplicate or two true neighbors. Both are retained and counted in the ambiguity diagnostic.
