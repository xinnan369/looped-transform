# LC-004A status

Completed 6 new 40-epoch runs, 6 matched RTX 3090 profiles and validation of the 24 archived LC-003 checkpoints. New runs used 5,400 measured optimizer updates each and 24.9 minutes of measured epoch training and validation. The stack starts at PR #8 head `d177916fde808ef5e00b0ca1b0220c4a01e8455e`.

| Task | Outcome and evidence |
|---|---|
| T1 | Full-depth results reproduced LC-003. Per-image and density-grouped prefix diagnostics, activation summaries and BN statistics are archived. [Diagnosis](loop_diagnosis.md) |
| T2 | Selected `train` BN for every stage experiment. [Ablation CSV](bn_ablation.csv) |
| T3 | Both gain arms trained for 40 epochs with matched fixed-window controls. [Experiment matrix](experiment_matrix.csv) · [Per-epoch gains](gain_trace.csv) |
| T4 | Additional seeds `not_run`. [Gate and paired seed file](paired_seed_results.csv) |
| T5 | 73 tests passed on each local Python 3.10, 3.11 and 3.13. [Tests and commands](test_results.md) · [Split epoch archives and hashes](training_history.json) |
| Resources | 0 external experiment bytes. [Resource audit](download_audit.md) |

## T2 · Backbone batch normalization

Frozen BN keeps running mean, variance and counters unchanged while training affine weights. Each frozen checkpoint's running buffers are bitwise equal to pretrained values. The train-mode arms reused the LC-003 seed-42 warm-up cosine runs. The evaluator reproduced their archived metrics, and preflight checked config, source SHA, manifest, weight, checkpoint, history and log provenance. Realized crop tallies and the full applied LR arrays matched the reused arms. All runs use batch two, BF16, random native 512 px crops and the same point target and optimizer.

The rule requires smaller late MAE for both CNN and Shared-2, a stability pass for both frozen arms and a decrease in late local nAP of at most 0.005 for each. Other outcomes select train-mode.

| Variant | BN | Last-five MAE | MAE CV | Late loss change | Last-five local nAP | Gate |
|---|---|---:|---:|---:|---:|:---:|
| cnn_only | train | 207.94 | 7.43% | -0.65% | 0.17716 | pass |
| cnn_only | frozen | 201.39 | 11.54% | -0.82% | 0.17146 | fail |
| shared_1 | train | 216.27 | 5.28% | -0.76% | 0.17613 | pass |
| shared_1 | frozen | 193.85 | 2.06% | -0.89% | 0.17749 | pass |
| shared_2 | train | 218.77 | 9.95% | -0.82% | 0.17648 | pass |
| shared_2 | frozen | 203.89 | 8.03% | -0.87% | 0.17826 | pass |
| unshared_2 | train | 213.12 | 6.46% | -0.85% | 0.18410 | pass |
| unshared_2 | frozen | 193.73 | 5.78% | -0.95% | 0.17766 | pass |

Frozen BN changed cnn_only late MAE by -6.55 and late nAP by -0.00570. Its MAE CV was 11.54%.

Frozen BN changed shared_2 late MAE by -14.89 and late nAP by +0.00178. Its MAE CV was 8.03%.

## T3 · Tied and stage-specific residual gains

Both gain models share the same WindowBlock, including all attention, FFN and layer-normalization weights. The attention and FFN residuals have separate g=2·sigmoid(a) factors. Tied gain uses two logits for both calls, and stage gain uses two per call. a=0 gives g=1 and preserves the original initial output and core gradients. Fixed windows are unchanged and both calls execute. The gain traces summarize internal validation tiles at every epoch.

| Variant | Last-five MAE / RMSE / local nAP | Best epoch and same metrics | Final metrics | MAE CV | Loss change | Stable |
|---|---|---|---|---:|---:|:---:|
| cnn_only | 207.94 / 310.09 / 0.17716 | E23 · 161.67 / 239.82 / 0.19085 | 206.30 / 312.71 / 0.18053 | 7.43% | -0.65% | pass |
| shared_1 | 216.27 / 338.49 / 0.17613 | E25 · 151.93 / 221.47 / 0.19861 | 224.67 / 356.50 / 0.17144 | 5.28% | -0.76% | pass |
| shared_2 | 218.77 / 342.18 / 0.17648 | E25 · 160.70 / 249.90 / 0.20344 | 218.03 / 342.09 / 0.17268 | 9.95% | -0.82% | pass |
| unshared_2 | 213.12 / 331.55 / 0.18410 | E19 · 140.80 / 204.42 / 0.19689 | 222.57 / 342.72 / 0.17505 | 6.46% | -0.85% | pass |
| shared_2_tied_gain | 219.27 / 341.46 / 0.17635 | E25 · 156.53 / 247.04 / 0.20424 | 221.13 / 347.44 / 0.17026 | 10.35% | -0.81% | fail |
| shared_2_stage_gain | 223.75 / 349.92 / 0.17244 | E25 · 157.93 / 251.80 / 0.20141 | 221.37 / 351.67 / 0.16770 | 9.42% | -0.79% | pass |

The fixed-prefix intervention holds weights and the head fixed. E40 was chosen before training as the snapshot for the contribution gate. Best-MAE checkpoint results remain available as secondary diagnostics.

| Variant | E40 MAE 0 / 1 / 2 calls | E40 local nAP 0 / 1 / 2 calls | Second minus first MAE | Improved / degraded / tied | E40 relative update 1 / 2 |
|---|---|---|---:|---|---|
| shared_2 | 191.27 / 372.73 / 218.03 | 0.15334 / 0.13828 / 0.17268 | -154.70 | 20 / 10 / 0 | 1.9505 / 0.9549 |
| shared_2_tied_gain | 201.73 / 338.93 / 221.13 | 0.15097 / 0.13480 / 0.17026 | -117.80 | 19 / 11 / 0 | 1.9013 / 0.9505 |
| shared_2_stage_gain | 193.73 / 350.73 / 221.37 | 0.15065 / 0.13326 / 0.16770 | -129.37 | 18 / 12 / 0 | 1.9276 / 0.9494 |

`shared_2_tied_gain` E40 attention/FFN gains were 0.9754/0.9841 at call one and 0.9754/0.9841 at call two.

`shared_2_stage_gain` E40 attention/FFN gains were 0.9815/0.9913 at call one and 0.9828/0.9774 at call two.

RTX 3090 profiles use the same resident 512 px batch-two inputs and precision. Full-frame latency includes tiled inference and candidate decoding on the same internal-validation image. FLOPs use the existing approximation for convolution, linear and attention matmuls. Four elementwise residual scaling operations for the gain models over a two-call tile add 3,145,728 scalar multiplications outside that estimate. Peak training memory reports allocated/reserved MiB.

| Variant | Parameters | GFLOP/tile | Batch-two step ms | Batch-one tile ms | Full image ms | Memory MiB |
|---|---:|---:|---:|---:|---:|---:|
| cnn_only | 1,151,750 | 25.095 | 10.93 | 2.09 | 45.91 | 358.7 / 452.0 |
| shared_1 | 1,596,614 | 28.920 | 13.19 | 2.99 | 52.71 | 439.5 / 484.0 |
| shared_2 | 1,596,614 | 32.745 | 14.84 | 3.93 | 53.58 | 498.5 / 538.0 |
| unshared_2 | 2,041,478 | 32.745 | 14.44 | 3.48 | 51.30 | 504.5 / 546.0 |
| shared_2_tied_gain | 1,596,616 | 32.745 | 40.92 | 5.07 | 99.88 | 508.9 / 678.0 |
| shared_2_stage_gain | 1,596,618 | 32.745 | 43.90 | 9.61 | 99.66 | 508.9 / 678.0 |

The sequential gain profiles were much slower than the ordinary Shared-2 profile. Other workstation CUDA processes were active after the queue, with sampled GPU utilization at 97%. A follow-up measured the three variants in rotating order within one process, one at a time, using the same inputs and nine repetitions. The paired measurements represent a workstation under concurrent load. Both sets of raw timing samples are preserved.

| Variant | Interleaved batch-two step ms | Interleaved tile ms | Interleaved full image ms |
|---|---:|---:|---:|
| shared_2 | 14.81 | 3.80 | 52.88 |
| shared_2_tied_gain | 14.86 | 4.13 | 53.23 |
| shared_2_stage_gain | 19.93 | 4.08 | 54.18 |

Rotating-order full-image medians ranged from 52.88 to 54.18 ms. Both tables describe their respective measurement conditions.

## T4 · Continuation gate

Stage gain must pass finite loss and gradient checks, ≤10% late MAE CV and ≤3% absolute change between E31–35 and E36–40 mean training loss. The primary endpoint must improve at least 3% over Shared-1, Shared-2 and tied gain. The best comparator's late nAP may lead by at most 0.005. At E40 call two must lower MAE relative to call one and improve more images than it degrades. Metrics use the fixed internal validation count cutoff and local nAP floor.

| Check | Observed | Result |
|---|---|:---:|
| Original stability gate | CV 9.42%; loss change -0.79%; finite True | pass |
| MAE reduction vs shared_2 | -2.27% | fail |
| MAE reduction vs shared_2_tied_gain | -2.04% | fail |
| MAE reduction vs shared_1 | -3.46% | fail |
| Local nAP − best comparator | -0.00403 | pass |
| E40 fixed-prefix contribution | -129.37 MAE; 18 improved / 12 degraded | pass |

The gate stopped seeds 123/2026 for this mechanism. The gain results remain a bounded single-seed pilot. In this experiment the second call helped its own trained checkpoint. Stage-dependent residual weights yielded higher late MAE than tied gain and both shared fixed-gain controls. The near-unit learned gains and the activation summaries describe how this run used the mechanism.

[Issue #9](https://github.com/xinnan369/looped-transform/issues/9) · [PR #8](https://github.com/xinnan369/looped-transform/pull/8) · [Next steps](next_steps.md)
