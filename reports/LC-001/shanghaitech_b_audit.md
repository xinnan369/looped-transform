# Point-candidate capacity audit (real)

Crop configuration uses one 512×512 crop per image with seed 42.
Counts and rates in the table are pooled within each dataset, split and mode.
The nearest-neighbor median pools within-image point distances.
Point loss rates divide lost points by the number of points in that row.

| Dataset | Split | Mode | Images | Source heads | Audited heads | Median W × H | Median heads/image | Median NN (px) | Stride | Occupied cells | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shanghaitech_b | all | full-image | 716 | 88272 | 88272 | 1024.00 × 768.00 | 95.50 | 16.17 | 4 | 0.2495% | 0.5268% | 0.0215% | 0.0000% | 0.0000% |
| shanghaitech_b | all | full-image | 716 | 88272 | 88272 | 1024.00 × 768.00 | 95.50 | 16.17 | 8 | 0.9629% | 4.0296% | 0.5392% | 0.0261% | 0.0000% |
| shanghaitech_b | all | crop-512 | 716 | 88272 | 18993 | 512.00 × 512.00 | 16.00 | 24.75 | 4 | 0.1615% | 0.2475% | 0.0000% | 0.0000% | 0.0000% |
| shanghaitech_b | all | crop-512 | 716 | 88272 | 18993 | 512.00 × 512.00 | 16.00 | 24.75 | 8 | 0.6348% | 1.9797% | 0.1580% | 0.0053% | 0.0000% |
| shanghaitech_b | test | full-image | 316 | 39121 | 39121 | 1024.00 × 768.00 | 93.00 | 16.50 | 4 | 0.2506% | 0.4933% | 0.0256% | 0.0000% | 0.0000% |
| shanghaitech_b | test | full-image | 316 | 39121 | 39121 | 1024.00 × 768.00 | 93.00 | 16.50 | 8 | 0.9718% | 3.5454% | 0.4780% | 0.0179% | 0.0000% |
| shanghaitech_b | test | crop-512 | 316 | 39121 | 8655 | 512.00 × 512.00 | 17.00 | 24.21 | 4 | 0.1667% | 0.2773% | 0.0000% | 0.0000% | 0.0000% |
| shanghaitech_b | test | crop-512 | 316 | 39121 | 8655 | 512.00 × 512.00 | 17.00 | 24.21 | 8 | 0.6547% | 2.0913% | 0.1155% | 0.0000% | 0.0000% |
| shanghaitech_b | train | full-image | 360 | 44856 | 44856 | 1024.00 × 768.00 | 101.50 | 15.68 | 4 | 0.2521% | 0.5484% | 0.0178% | 0.0000% | 0.0000% |
| shanghaitech_b | train | full-image | 360 | 44856 | 44856 | 1024.00 × 768.00 | 101.50 | 15.68 | 8 | 0.9685% | 4.4877% | 0.6086% | 0.0312% | 0.0000% |
| shanghaitech_b | train | crop-512 | 360 | 44856 | 9402 | 512.00 × 512.00 | 15.50 | 24.99 | 4 | 0.1590% | 0.2234% | 0.0000% | 0.0000% | 0.0000% |
| shanghaitech_b | train | crop-512 | 360 | 44856 | 9402 | 512.00 × 512.00 | 15.50 | 24.99 | 8 | 0.6248% | 2.0102% | 0.2127% | 0.0106% | 0.0000% |
| shanghaitech_b | val | full-image | 40 | 4295 | 4295 | 1024.00 × 768.00 | 88.00 | 17.75 | 4 | 0.2171% | 0.6054% | 0.0233% | 0.0000% | 0.0000% |
| shanghaitech_b | val | full-image | 40 | 4295 | 4295 | 1024.00 × 768.00 | 88.00 | 17.75 | 8 | 0.8419% | 3.6554% | 0.3725% | 0.0466% | 0.0000% |
| shanghaitech_b | val | crop-512 | 40 | 4295 | 936 | 512.00 × 512.00 | 16.50 | 28.33 | 4 | 0.1425% | 0.2137% | 0.0000% | 0.0000% | 0.0000% |
| shanghaitech_b | val | crop-512 | 40 | 4295 | 936 | 512.00 × 512.00 | 16.50 | 28.33 | 8 | 0.5676% | 0.6410% | 0.0000% | 0.0000% | 0.0000% |
