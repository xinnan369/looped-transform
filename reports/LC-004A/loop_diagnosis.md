# LC-004A fixed-prefix diagnosis

The diagnosis reuses the six best/last Shared-2 snapshots and the 18 matched control snapshots from LC-003. The 24 files represent 12 trained runs across seeds 42, 123 and 2026. SHA-256 verified the manifest, ResNet18 weights, each checkpoint and the archived run histories and logs. Every full-depth evaluation reproduced the archived count metrics and local nAP. Each full Shared-2 prefix produced bitwise identical decoded candidates, tile IDs and ownership flags to the ordinary forward path on all 30 validation images.

For each Shared-2 snapshot, 0, 1 and 2 calls feed the same trained head. Prefix results measure interventions on one trained checkpoint. Only full-depth arms were independently trained. All prefixes use the same 30 images, 14,177 GT heads, 0.30 count cutoff, candidate ownership and 0.05 local nAP score floor. No image or label determines the number of calls.

## Whole-split paired comparison

| Seed | Snapshot (epoch) | MAE calls 0 / 1 / 2 | RMSE calls 0 / 1 / 2 | Local nAP calls 0 / 1 / 2 | Calls 2−1 MAE | Improved / degraded / tied images |
|---:|---|---|---|---|---:|---|
| 42 | best (25) | 188.83 / 312.93 / 160.70 | 282.97 / 432.73 / 249.90 | 0.1535 / 0.1498 / 0.2034 | -152.23 | 21 / 9 / 0 |
| 42 | last (40) | 191.27 / 372.73 / 218.03 | 312.96 / 530.18 / 342.09 | 0.1533 / 0.1383 / 0.1727 | -154.70 | 20 / 10 / 0 |
| 123 | best (28) | 210.27 / 337.47 / 172.03 | 347.62 / 462.55 / 270.70 | 0.1527 / 0.1416 / 0.1969 | -165.43 | 20 / 9 / 1 |
| 123 | last (40) | 213.53 / 445.43 / 252.20 | 354.67 / 640.78 / 402.03 | 0.1499 / 0.1283 / 0.1779 | -193.23 | 23 / 7 / 0 |
| 2026 | best (23) | 164.83 / 368.03 / 183.70 | 244.94 / 533.67 / 274.83 | 0.1580 / 0.1626 / 0.1945 | -184.33 | 17 / 13 / 0 |
| 2026 | last (40) | 205.30 / 410.20 / 285.43 | 312.70 / 611.33 / 440.38 | 0.1582 / 0.1530 / 0.1704 | -124.77 | 21 / 8 / 1 |

The calls 2−1 difference holds the trained head and checkpoint fixed. The 0- and 1-call heads see intermediate features that training did not directly supervise. A favorable within-checkpoint effect can coexist with higher error than an independently trained one-call model.

Relative to one call, two calls improved both whole-split MAE and local nAP in all six snapshots. Relative to zero calls, two calls improved local nAP in all six snapshots and MAE in two. These comparisons reveal distinct effects on fixed-cutoff counting and score-ranked localization.

## Density and local matching

Training-only density tertiles are 545.60 and 1080.58 heads/Mpx, as in LC-003. Groups are closed on the upper boundary. Group AP pools candidate ranks within the group. Each row below compares two calls with one call in the same checkpoint and GT group.

| Seed | Snapshot | Density | Images | MAE 1 → 2 | nAP 1 → 2 | Improved / degraded / tied |
|---:|---|---|---:|---|---|---|
| 42 | best | low | 11 | 253.64 → 63.91 | 0.1259 → 0.1658 | 9 / 2 / 0 |
| 42 | best | mid | 8 | 359.25 → 181.62 | 0.1484 → 0.1816 | 5 / 3 / 0 |
| 42 | best | high | 11 | 338.55 → 242.27 | 0.1855 → 0.2321 | 7 / 4 / 0 |
| 42 | last | low | 11 | 304.09 → 95.91 | 0.1195 → 0.1408 | 9 / 2 / 0 |
| 42 | last | mid | 8 | 456.88 → 258.12 | 0.1395 → 0.1535 | 6 / 2 / 0 |
| 42 | last | high | 11 | 380.18 → 311.00 | 0.1701 → 0.1961 | 5 / 6 / 0 |
| 123 | best | low | 11 | 220.18 → 61.82 | 0.1159 → 0.1624 | 8 / 2 / 1 |
| 123 | best | mid | 8 | 375.00 → 206.38 | 0.1352 → 0.1739 | 6 / 2 / 0 |
| 123 | best | high | 11 | 427.45 → 257.27 | 0.1774 → 0.2288 | 6 / 5 / 0 |
| 123 | last | low | 11 | 331.91 → 118.73 | 0.1068 → 0.1394 | 8 / 3 / 0 |
| 123 | last | mid | 8 | 542.12 → 317.12 | 0.1188 → 0.1395 | 7 / 1 / 0 |
| 123 | last | high | 11 | 488.64 → 338.45 | 0.1648 → 0.2120 | 8 / 3 / 0 |
| 2026 | best | low | 11 | 229.64 → 86.27 | 0.1149 → 0.1404 | 5 / 6 / 0 |
| 2026 | best | mid | 8 | 449.50 → 274.75 | 0.1323 → 0.1618 | 4 / 4 / 0 |
| 2026 | best | high | 11 | 447.18 → 214.91 | 0.2104 → 0.2323 | 8 / 3 / 0 |
| 2026 | last | low | 11 | 214.82 → 136.82 | 0.1044 → 0.1218 | 6 / 4 / 1 |
| 2026 | last | mid | 8 | 469.25 → 289.12 | 0.1221 → 0.1445 | 6 / 2 / 0 |
| 2026 | last | high | 11 | 562.64 → 431.36 | 0.1940 → 0.2010 | 9 / 2 / 0 |

The checkpoint JSON and per-image CSV record observed and GT counts, count bias, local TP/FP/FN, score histograms and near-neighbor competition. TP/FP/FN use greedy one-to-one matching at δ=0.50 on owned candidates with score >0.30. They are protocol-local. AP ranks all in-frame tile candidates above 0.05. A 2 px pair with two distinct accepted matches is counted as two matched neighbors. Other close pairs are unresolved geometry. Local competition counts GT with multiple eligible candidates and unmatched predictions within any GT radius. Neither proximity alone nor competition identifies true duplicates.

All six one-call prefixes overcounted the fixed total of 14,177 heads, predicting 18,169–23,711. Call two reduced local FP and the number of GT with competing eligible predictions in every snapshot. Local FN rose in five snapshots even as nAP improved. The matched counts distinguish a reduction in false positives at the fixed score from improvement in score-ranked AP.

## Token updates and BN buffers

| Seed | Snapshot | Relative L2 update 1 / 2 | Cosine 1 / 2 | Attention RMS 1 / 2 | FFN RMS 1 / 2 |
|---:|---|---|---|---|---|
| 42 | best | 1.9984 / 0.9537 | 0.3978 / 0.7130 | 0.2526 / 0.3532 | 0.2752 / 0.2049 |
| 42 | last | 1.9505 / 0.9549 | 0.4016 / 0.7186 | 0.2392 / 0.3307 | 0.2608 / 0.2033 |
| 123 | best | 1.9619 / 0.9150 | 0.4473 / 0.7497 | 0.2544 / 0.3339 | 0.2616 / 0.1993 |
| 123 | last | 1.9080 / 0.9135 | 0.4485 / 0.7540 | 0.2379 / 0.3090 | 0.2509 / 0.1980 |
| 2026 | best | 1.9111 / 0.9072 | 0.4387 / 0.7963 | 0.2599 / 0.3461 | 0.2573 / 0.1904 |
| 2026 | last | 1.7974 / 0.8747 | 0.4460 / 0.7983 | 0.2523 / 0.3131 | 0.2472 / 0.1854 |

Call two had a smaller relative update and higher cosine in all six snapshots. Its attention residual RMS increased in all six, while FFN residual RMS decreased. These are changes in the scale and composition of the update, measured on frozen checkpoints.

RMS, flattened cosine and relative L2 use float32 reductions. The denominator is the input L2 plus 1e−12. Spatial variation is mean channel-wise population SD across tokens. Channel variation is mean token-wise population SD across channels. Results equally weight all 512 px inference tiles, including the backbone token grid formed from padded tiles. The checkpoint files contain both input and output variation, residual RMS and tile counts.

Per-layer BN running mean and variance snapshots report q05, median, q95, RMS and deviation from the pretrained buffers. The current evaluations observe inference activations. The LC-003 archives contain scalar training histories and checkpoint buffers, so they cannot recover training-time token or residual trajectories. Activation changes are descriptive measurements and leave the source of the accuracy change unresolved.

[Checkpoint index, grouped metrics and per-image CSV](loop_diagnosis.json)
