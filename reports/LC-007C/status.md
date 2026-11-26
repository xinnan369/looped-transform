# LC-007C stagewise and endpoint-gradient diagnosis

**The CPU scope and separately approved GPU diagnosis completed.** This draft stacks on accepted LC-007B PR #20 head `0aeaea0562f075fd28ecaa48c6400193a7cf22f4`. The CPU freeze is `ad0f4840c37f11a948fa43951d0d672351a6eb6d`. The GPU runner and unchanged numerical tolerances froze at `2f951ed843669bca3cc614e296cec1230c750430`.

The completed GPU scope received a technical pass. The original approval preceded execution at 08:30:55 Asia/Shanghai on November 26 and required the background-job check. The draft remains open for owner acceptance. [Authorization record](authorization_record.md)

| Deliverable | Evidence |
| --- | --- |
| Archive-only diagnosis | All 24 indexed files and 42,560 rows reconciled to frozen checksums and sizes. Recomputed 80 seed/epoch/policy/group comparisons, 48 endpoint summaries and 800 paired-image rows. [Stagewise analysis](stagewise_analysis.md) · [Archive integrity](archive_integrity.json) |
| Train-only preparation | Four E40 checkpoint digests verified. Fixed 12 batches per seed, each with two crops, and four batches per density group, yielding the 96 project cases. [Frozen probe manifest](probe_manifest.json) |
| Gradient measurement | 96 cases completed with exact output/loss agreement, a maximum gradient reconciliation difference of 0.000732422, and checkpoint and state invariance. 480 defined and 144 unavailable directions. [Execution](gpu_execution.json) · [Equivalence](gradient_equivalence.json) · [Measured gradients](gradient_diagnostics.csv) |
| Mechanism decision | No negative cosine in any of the 240 defined shared-parameter comparisons. The second-call nAP deficit remains unresolved. The results do not motivate a conflict intervention. [Gradient analysis](gradient_analysis.md) · [Hypothesis register](hypothesis_register.md) |

On EMA E36–40 at high density, Shared-2 aux starts 4.36 MAE behind equally supervised Unshared-2. Its stronger average second-call count correction reduces the final gap to 1.77, combining opposite seed-level effects. Its stagewise local nAP gain is smaller in both seeds and both policies in the high-density and overall strata. The archive cannot identify gradients or specific recovered heads.

The GPU phase took 13.4148 synchronized seconds and peaked at 818 MiB process reserved. The separate background training process remained active. Available device memory allowed completion under the process cap. No project training, official test access, new downloads or hosted CI took place. [Approved execution](gpu_execution_protocol.md) · [Resource audit](resource_audit.md) · [Local tests](test_results.md) · [Handoff](next_steps.md)
