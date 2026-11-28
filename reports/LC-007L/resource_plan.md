# One-attempt resource envelope

Proposed one RTX 3090 case, 40 epochs, 7,200 successful updates, 1,800 charged seconds, zero disposable GPU updates and zero external downloads. The launcher sends TERM at 1,795 s and allows five seconds before hard kill. CPU preflight, device checks, training, validation, transfers, checkpoints, logging, gaps, failures and cleanup count toward the total.

| Phase | S768 seconds | U768 seconds | Planned seconds |
| --- | ---: | ---: | ---: |
| Training | 578.415 | 643.027 | 643.027 |
| Live validation | 296.558 | 307.634 | 307.634 |
| EMA validation | 284.369 | 319.470 | 319.470 |
| Initialization | .902 | .146 | .902 |
| Checkpoint | 1.751 | 2.861 | 2.861 |
| Phasewise base | | | 1,273.894 |
| 25% plus 120 s allowance | | | 438.473 |
| Total projection | | | 1,712.367 |

Use the phasewise maximum of the observed historical S768 and U768 timings. Their charged totals were 1,174.758 s and 1,283.194 s, with peak reserved 1,656 MiB and 1,686 MiB. The allowance covers setup, the small affine instrumentation, device contention and unknown execution differences. The estimate leaves 87.6 seconds within the proposed hard ceiling. Actual candidate GPU time and memory remain unmeasured.

At a separately authorized launch, check at least 3 GiB available disk and physical device headroom against the 3,535,798,272-byte estimate, two times the larger historical peak reserved. This is a preflight availability check. The runner observes actual device memory and process peaks without configuring a quota or allocator fraction, and leaves concurrent jobs running. On OOM, nonfinite, cache, time or trajectory failure, stop and retain the partial evidence without retry or changes to batch or precision.

The preparation snapshot reported 95,445,192,704 bytes free on WSL disk and 19,249 MiB free on the RTX 3090. A background process was present. Free memory and concurrent work are checked again at launch.

Issue #39 requires independent prelaunch review and exact owner approval. Its authorization binds the source, tree, config and limits. Previous waivers or direct launch records do not transfer. [Source and scope](protocol.md)
