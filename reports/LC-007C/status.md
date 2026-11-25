# LC-007C CPU diagnosis and proposed GPU probe

**CPU scope complete. Project GPU diagnostics are pending separate approval.** This draft stacks on accepted LC-007B PR #20 head `0aeaea0562f075fd28ecaa48c6400193a7cf22f4`. The CPU freeze is `ad0f4840c37f11a948fa43951d0d672351a6eb6d`.

| Deliverable | Evidence |
| --- | --- |
| Archive-only diagnosis | All 24 indexed files and 42,560 rows reconciled to frozen checksums and sizes. Recomputed 80 seed/epoch/policy/group comparisons, 48 endpoint summaries and 800 paired-image rows. [Stagewise analysis](stagewise_analysis.md) · [Archive integrity](archive_integrity.json) |
| Train-only preparation | Four E40 checkpoint byte digests verified. Fixed 12 batches per seed, each with two crops, and four batches per density group. The future matrix contains at most 96 checkpoint/batch/policy cases. [Frozen probe manifest](probe_manifest.json) |
| Gradient measurement method | Synthetic CPU FP32 and BF16 forward/loss equivalence, block and decoder call-gradient sum checks, undefined cosines, restoration and cooperative resource guard tests. Project gradients remain `not_run`. [Synthetic evidence](gradient_equivalence.json) · [Project status](gradient_diagnostics.csv) |
| Mechanism decision | Inconclusive. Count correction and localization ranking provide different evidence. Keep the shared-call and early-vs-final supervision hypotheses separate. [Hypothesis register](hypothesis_register.md) |

On EMA E36–40 at high density, Shared-2 aux starts 4.36 MAE behind equally supervised Unshared-2. Its stronger average second-call count correction reduces the final gap to 1.77, combining opposite seed-level effects. Its stagewise local nAP gain is smaller in both seeds and both policies in the high-density and overall strata. The archive cannot identify gradients or specific recovered heads.

The proposed GPU budget is 1,200 synchronized seconds and 2 GiB process peak reserved. Actual GPU time, allocator usage and project gradient observations are absent. No real project training, checkpoint deserialization, official test access or new downloads were performed. [Protocol](protocol.md) · [Resource audit](resource_audit.md) · [Local tests](test_results.md) · [Handoff](next_steps.md)
