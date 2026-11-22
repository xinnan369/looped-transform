# Point-candidate capacity audit (real)

Crop configuration uses one 512×512 crop per image with seed 42.
Counts and rates in the table are pooled within each dataset, split and mode.
The nearest-neighbor median pools within-image point distances.
Point loss rates divide lost points by the number of points in that row.

| Dataset | Split | Mode | Images | Source heads | Audited heads | Median W × H | Median heads/image | Median NN (px) | Stride | Occupied cells | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shanghaitech_a | all | full-image | 482 | 241275 | 241275 | 1024.00 × 647.00 | 359.00 | 11.10 | 4 | 1.4734% | 0.8932% | 0.0381% | 0.0017% | 0.0000% |
| shanghaitech_a | all | full-image | 482 | 241275 | 241275 | 1024.00 × 647.00 | 359.00 | 11.10 | 8 | 5.4769% | 7.5685% | 1.4689% | 0.0829% | 0.0017% |
| shanghaitech_a | all | crop-512 | 482 | 241275 | 135446 | 512.00 × 512.00 | 208.50 | 10.69 | 4 | 1.6987% | 0.9591% | 0.0354% | 0.0000% | 0.0000% |
| shanghaitech_a | all | crop-512 | 482 | 241275 | 135446 | 512.00 × 512.00 | 208.50 | 10.69 | 8 | 6.3134% | 7.9759% | 1.5741% | 0.0812% | 0.0000% |
| shanghaitech_a | test | full-image | 182 | 78862 | 78862 | 1024.00 × 611.00 | 307.50 | 11.93 | 4 | 1.3158% | 0.7786% | 0.0355% | 0.0051% | 0.0000% |
| shanghaitech_a | test | full-image | 182 | 78862 | 78862 | 1024.00 × 611.00 | 307.50 | 11.93 | 8 | 4.9545% | 6.2210% | 1.2186% | 0.0647% | 0.0051% |
| shanghaitech_a | test | crop-512 | 182 | 78862 | 45831 | 512.00 × 512.00 | 201.50 | 11.54 | 4 | 1.5261% | 0.7091% | 0.0240% | 0.0000% | 0.0000% |
| shanghaitech_a | test | crop-512 | 182 | 78862 | 45831 | 512.00 × 512.00 | 201.50 | 11.54 | 8 | 5.7534% | 6.4171% | 1.1346% | 0.0436% | 0.0000% |
| shanghaitech_a | train | full-image | 270 | 148236 | 148236 | 1024.00 × 661.50 | 378.00 | 10.68 | 4 | 1.5572% | 0.9822% | 0.0432% | 0.0000% | 0.0000% |
| shanghaitech_a | train | full-image | 270 | 148236 | 148236 | 1024.00 × 661.50 | 378.00 | 10.68 | 8 | 5.7456% | 8.3596% | 1.6656% | 0.0978% | 0.0000% |
| shanghaitech_a | train | crop-512 | 270 | 148236 | 80129 | 512.00 × 512.00 | 217.50 | 10.33 | 4 | 1.7910% | 1.1232% | 0.0462% | 0.0000% | 0.0000% |
| shanghaitech_a | train | crop-512 | 270 | 148236 | 80129 | 512.00 × 512.00 | 217.50 | 10.33 | 8 | 6.6026% | 8.8732% | 1.9019% | 0.1111% | 0.0000% |
| shanghaitech_a | val | full-image | 30 | 14177 | 14177 | 800.00 × 508.00 | 362.00 | 10.81 | 4 | 1.6460% | 0.5996% | 0.0000% | 0.0000% | 0.0000% |
| shanghaitech_a | val | full-image | 30 | 14177 | 14177 | 800.00 × 508.00 | 362.00 | 10.81 | 8 | 6.1492% | 6.7927% | 0.8041% | 0.0282% | 0.0000% |
| shanghaitech_a | val | crop-512 | 30 | 14177 | 9486 | 512.00 × 512.00 | 210.00 | 9.81 | 4 | 1.9149% | 0.7801% | 0.0000% | 0.0000% | 0.0000% |
| shanghaitech_a | val | crop-512 | 30 | 14177 | 9486 | 512.00 × 512.00 | 210.00 | 9.81 | 8 | 7.1077% | 7.9275% | 0.9277% | 0.0105% | 0.0000% |
