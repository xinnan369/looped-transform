# Historical stability diagnosis

The [run ledger](run_ledger.csv) separates 17 LC-003, six LC-004A, six LC-005 and 18 LC-006 fresh trajectories. The six LC-004A runs comprise four frozen-BN runs and two gain runs. The other 12 LC-004A training records point to the original LC-003 trajectories. LC-003 replayed seven LC-002R best snapshots. LC-004A replayed 24 LC-003 and four gain snapshots and inspected eight frozen-BN checkpoint buffer records. LC-004B contains 18 checkpoint/call combinations from 12 E40 checkpoint files, including six intermediate-prefix cases. Its 60-image fit and validation replay did not add training trajectories. Historical disposable preflight and timing steps in LC-004A/005/006 belong to their respective profile artifacts and are outside the 47-run count.

| Archive | Fresh trajectories | Tracked experiment updates | Historical stability passes |
| --- | ---: | ---: | ---: |
| LC-003 | 17 | 91,800 | 11/17 |
| LC-004A | 6 | 32,400 | See the frozen-BN and gain decisions |
| LC-005 | 6 | 32,400 | 6/6 |
| LC-006 | 18 | 129,600 | 4/18 |
| Total | 47 | 286,200 | |

The [recomputation](historical_recomputation.csv) derives the late mean MAE, RMSE and local nAP, population CV and signed E31–35 to E36–40 final-loss change from 1,640 archived epochs. It checks the saved numbers and decisions with absolute and relative tolerance `1e-12`. [Paired late effects](historical_paired_late.csv) preserve seed, comparison, dataset and stage; the LC-003 constant/cosine pilot is seed 42 only. The primary window and all historical decisions retain their original definitions.

| Dataset / archive | Training arm | E36–40 MAE | RMSE | Local nAP |
| --- | --- | ---: | ---: | ---: |
| A / LC-003 cosine | CNN | 220.55 | 328.43 | .17571 |
| A / LC-003 cosine | Shared-1 | 221.60 | 343.84 | .18129 |
| A / LC-003 cosine | Unshared-2 λ0 | 221.57 | 351.13 | .17952 |
| A / LC-005 | Shared-2 λ0 | 234.14 | 364.29 | .18108 |
| A / LC-005 | Shared-2 λ.25 | 221.23 | 348.41 | .17932 |
| B / LC-006 | CNN | 28.28 | 37.71 | .27479 |
| B / LC-006 | Shared-1 | 25.42 | 33.68 | .28956 |
| B / LC-006 | Shared-2 λ0 | 26.715 | 35.43 | .29230 |
| B / LC-006 | Shared-2 λ.25 | 25.765 | 34.56 | .29254 |
| B / LC-006 | Unshared-2 λ0 | 24.985 | 33.58 | .29474 |
| B / LC-006 | Unshared-2 λ.25 | 24.695 | 33.51 | .29647 |

On A, the matched LC-005 auxiliary loss reduced late count error by 12.90 heads (5.51%) and RMSE by 15.87, with local nAP down .001758. The early output improved from 388.82 to 219.09 MAE and had slightly lower error than its own aux final output. Mean count-mask FP declined by 296.9 and TP by 75.0 per validation epoch alongside a 75.0 rise in FN. The A comparison with LC-003 is cross-archive context. LC-003's Unshared-2 used a single readout in training and A has no equally supervised independent arm.

On B, the shared auxiliary gain was 0.95 heads (3.56%) and the independent auxiliary gain was 0.29 (1.16%). The difference of gains was 0.66 heads with paired seed sample SD 1.42. The shared aux mean was 0.34 heads above Shared-1 and 1.07 above matched Unshared-2 aux. Relative to shared λ0, mean count-mask TP rose by 14.9, FP by 3.4 and FN fell by 14.9. The shared aux late final readout improved over its own early MAE and local nAP in all three seeds. This within-checkpoint effect and the comparison with an independently trained one-call model answer different questions. [LC-005 decision](../LC-005/status.md) · [LC-006 decision](../LC-006/research_decision.json)

## Why an intervention should first target general stability

All 18 B final-loss changes were between −2.36% and −1.02%, with finite logged losses/gradients and zero AMP skips. The late final MAE population CV spanned 6.75–21.02%. Fourteen instability failures came from the CV bound, including all CNN and Shared-1 runs. CNN's CV was 11.10%, 12.65% and 12.08%. All six matched A auxiliary-graph runs passed, with CV 3.70–9.95%. A/B differ in density, count scale, step horizon and validation population, so comparing their relative dispersion cannot isolate a single cause.

For B, calculate Pearson correlations within each run (18 separate sets of five epochs). The median and observed range are descriptive. All-epoch values in the CSV include the common initial-learning trend.

| B E36–40 signal paired with final MAE | Median r | Range |
| --- | ---: | ---: |
| Predicted total | −.981 | −.998 to +.532 |
| TP / FN | −.988 / +.988 | −.999 to +.473 / −.473 to +.999 |
| FP | −.958 | −.997 to +.559 |
| Local nAP | −.442 | −.898 to +.750 |
| Final classification loss | −.508 | −.974 to +.438 |
| Final offset loss | −.277 | −.678 to +.531 |
| Sampled GT heads | +.212 | −.637 to +.581 |
| Mean gradient norm | +.014 | −.766 to +.804 |

All B arms averaged fewer counted predictions than the 4,295 validation GT heads over the late window (3,373–3,754). TP and FN are mechanically linked by fixed GT, and FP can rise together with recall. The count total tracks several late excursions more closely than loss or mean gradient norm. Since LC-003/005/006 training logs lack per-epoch per-image inference, the concentration of B epoch-to-epoch MAE swings is unavailable. The fixed E40 top-five contributions and image resampling address sensitivity at one snapshot.

## Implementation audit

| Path | Evidence and interpretation |
| --- | --- |
| Point assignment and losses | The code sorts native crop points by `(x,y)`, floors to stride-four cells and assigns two slots. Overflow remains in `heads` and density counts, and is excluded from the slot positives and offset targets. Focal BCE weights all valid slots and smooth L1 weights positive coordinates, both divided by the number of positives clamped to one over the batch. Fixed .25/.75 focal weights and the positive denominator make foreground prevalence relevant. Valid padding cells are masked. Density loss was disabled. [Targets](../../src/loopcrowd/targets.py) |
| Crop sampling | One seed/epoch/record-dependent random native 512 crop per training image. A late epoch means ranged 79,863–80,429 sampled heads and observed late overflow reached 43 heads per epoch. B means ranged 10,394–10,763 with late overflow reaching five. The arms in LC-005/006 used matched realized crop orders per seed. Whole-frame evaluation mosaics overlapping tiles, and the object mix in crops can differ from the ownership regions. B train and validation have 8 and 1 fixed-origin grid overflow points. [Crop loader](../../src/loopcrowd/data.py) · [B audit](../LC-006/data_audit.json) |
| Decoding and matcher | In-frame candidates above the raw floor are kept for local nAP, including overlapping non-owner candidates. Count keeps half-open midpoint owners above .30. The matcher sorts predictions stably by score, chooses the nearest available GT by spacing-normalized distance and requires strict distance below δ. AP pools ranked matches over images. Seam duplication and losses of eligible candidates are possible under this protocol and are recorded in the new tables. [Inference](../../src/loopcrowd/inference.py) · [Matcher](../../src/loopcrowd/metrics.py) |
| Shared graph | Shared-2 reuses the same attention/FFN block object twice. Both calls contribute gradients through the second-stage objective, and the auxiliary objective has its own gradient through the first call plus the common point head. Unshared-2 trains separate blocks and the same point head on both outputs. The synthetic gradients verify identical final-only behavior and an unchanged second independent block gradient when only the early loss weight changes. [Network](../../src/loopcrowd/model/__init__.py) · [Gradient fixtures](../../tests/test_intermediate_supervision.py) |
| BN buffers | Training mode updates backbone BN running mean/variance at batch size two. Evaluation restores the historical buffers and uses eval mode. LC-004A seed-42 frozen BN improved late MAE by 6.55 for CNN and 14.89 for Shared-2. Frozen CNN had CV 11.54% and nAP −.00570 versus train-mode. The original BN gate selected train mode, and there is no equivalent B ablation. [BN evidence](../LC-004A/status.md) |
| Optimizer and actual LR | AdamW uses weight decay .01 and a fixed update horizon for cosine. Each dataset's historical run archive records warmup, successful updates and actual LR extrema. A warmed up 270/5,400 and B 360/7,200 updates to .0002 then reached .00002. LC-005 and LC-006 verified equal applied LR trajectories within seed. This makes a differing assigned LR schedule an unlikely explanation for their paired arm deltas. LC-003 also recorded three constant-LR runs, separated as schedule contrasts. [Training](../../src/loopcrowd/training.py) · [B frozen protocol](../LC-006/protocol.md) |

No reproducible implementation defect was identified in these inspected paths. The frozen protocols involve measurable trade-offs in capacity, threshold retention and assignment. No training or historical metric code was changed.
