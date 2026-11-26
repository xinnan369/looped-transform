# LC-007C E40 gradient diagnosis

The approved RTX 3090 run completed all 96 fixed batch-policy cases after two generated BF16 preflight checks. It used the train-only crops, four local E40 checkpoints and two readouts with the same live BN buffers. All native and expanded outputs and losses were numerically equal. The largest absolute element difference in the reconciled block/decoder gradients was 0.000732422, within frozen BF16 absolute-plus-relative tolerances. No project parameter gradients, parameter values, buffer values, EMA, optimizer or scheduler state, RNG or checkpoint bytes changed. [Case evidence](gradient_diagnostics.csv) · [Equivalence](gradient_equivalence.json) · [Execution](gpu_execution.json)

## Evidence on a shared parameter basis

Each denominator in the table covers 12 batches in each of two seeds. The high-density subset contains four batches per seed. Rows involving different objectives share the same case inputs.

| Shared-2 comparison | Live cosine median | EMA cosine median | High-density live / EMA median | Negative live / EMA |
| --- | ---: | ---: | ---: | ---: |
| Block, early vs final supervision | .686 | .684 | .651 / .674 | 0/24 / 0/24 |
| Decoder, early vs final supervision | .874 | .856 | .854 / .827 | 0/24 / 0/24 |
| Block, call 1 vs call 2, final loss | .177 | .185 | .168 / .191 | 0/24 / 0/24 |
| Block, call 1 vs call 2, total loss | .197 | .200 | .188 / .211 | 0/24 / 0/24 |
| Decoder, call 1 vs call 2, total loss | .874 | .856 | .854 / .827 | 0/24 / 0/24 |

The final loss reaches the first block through call two. It has no path through the first decoder output, so the final-loss decoder call cosine is undefined. For the decoder total loss, the call-one vector is the weighted early decoder gradient and the call-two vector is the final decoder gradient. That gives the same cosine as the supervision comparison above.

All 240 defined Shared-2 comparisons had nonnegative cosine, including 80 in the high-density subset. All observed shared-block call cosines were strictly positive, with minimum .0745 for final loss and .0846 for total loss. The call vectors aligned less strongly than the two supervision vectors. The median call-one/call-two block norm ratios for final loss were 1.66 live and 1.70 EMA. For total loss they were 1.97 and 2.05. The λ-weighted early/final block norm ratios were .201 and .205. These measurements describe relative vector length and direction at the saved point.

Shared high-density final-loss call cosine medians were .1465/.1658 (live/EMA) for seed 42 and .1974/.1908 for seed 123. All four batches in every seed/policy/density cell had positive cosine for both the recurrence and supervision comparisons. [Seed, policy and density distributions](gradient_summary.csv)

## Independently parameterized reference

The Unshared-2 aux first block aligned with the final-loss direction under early supervision (medians .763 live and .777 EMA), as did its shared decoder (.880 and .871). Its second block has no early-loss path. The unshared final-loss block call cosines had medians .00375/.00627 (live/EMA), and total-loss medians .00399/.00690. There were 23 negative rows in those cross-parameter comparisons, all confined to the unshared blocks. They describe two different parameter spaces and do not show that one update opposed another on a tied tensor.

Across all 624 rows, 480 defined a cosine and 144 had no gradient path. No reachable pair fell under the near-zero limit and no nonfinite case was observed. The 144 missing directions are the first decoder under final loss in both arms (96) and early supervision of the second independent block (48). The evidence preserves each denominator explicitly.

## Decision

The E40 probes do not support opposing shared gradient directions as the explanation for the archived nAP correction gap. The archived counts already show an average first-stage high-density deficit and a shared second call that closes part of it, with opposite seed directions. The shared model's smaller nAP increment in both seeds and policies remains unresolved. Positive batch-level gradient directions at E40 leave open finer-grained interactions and effects earlier in training. Do not select an intervention to resolve gradient conflict from these results. A future diagnostic could compare candidate ranking, localization displacement and matched GT identity transitions under the same thresholds to distinguish the remaining explanations. [Stagewise archive](stagewise_analysis.md) · [Hypotheses](hypothesis_register.md)
