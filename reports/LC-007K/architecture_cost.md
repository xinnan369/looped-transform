# Matched geometry and compute

The U768 case recreates all four distinct historical U512 block hashes, plus the full, common, encoder, decoder and post-init RNG hashes. Common tensors and the first block match the accepted S768 initialization. Other U blocks keep their original distinct initialization. U uses 2,931,206 parameters and S uses 1,596,614. The selected parameter organization is the changed factor within the 768 comparison.

| Per-image forward path | Shared dense FLOPs | Unshared dense FLOPs |
| --- | ---: | ---: |
| 512 training-shaped, two reads | 55,677,288,448 | 55,677,288,448 |
| 768 training-shaped, two reads | 124,690,366,464 | 124,690,366,464 |
| Fixed 512 evaluation, four reads | 84,743,815,168 | 84,743,815,168 |
| Fixed 512 deployment, final only | 41,144,025,088 | 41,144,025,088 |

CPU synthetic hooks count dense convolution, linear and attention operations. The matched sizes have equal nominal dense forward cost. The 768 two-read forward costs 2.2395× the 512 forward, and its valid pixel area is 2.25×. The 768 pair each uses 8,493,465,600 valid pixels, 1,395,609 sampled GT and 180 over-slot annotations across 40 epochs. Backward, wall time and peak memory are measured only in an authorized run. [Initialization, exposure and compute proof](initialization_audit.json)
