# Initialization and architecture cost

CPU initialization reproduced the full model, common, encoder, decoder, block and post-init RNG hashes and 1,596,614 parameters of archived seed-42 S4-shift. Only `train_crop_size` is added to the reference recipe. Configurations without this field keep their inherited tile size, digest and three-field crop identity. Checkpoint resume rejects changing or dropping the explicit crop size. The 512 evaluation computation remains unchanged.

| Path | Dense forward FLOPs per image |
| --- | ---: |
| Historical 512 training-shaped two reads | 55,677,288,448 |
| Proposed 768 training-shaped two reads | 124,690,366,464 |
| Fixed 512 evaluation four reads | 84,743,815,168 |

CPU synthetic profiling counts convolution, linear and attention FLOPs. Larger-crop two-read dense cost is 2.2395× before backward, optimizer, data movement and bookkeeping. Window padding and the mixture of operators account for the difference from the 2.25× pixel-area factor. Timing and device peaks will be measured only if the frozen experiment runs. [Cost audit](initialization_audit.json)
