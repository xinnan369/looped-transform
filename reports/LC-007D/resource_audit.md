# Executed resource audit

The approved RTX 3090 screen completed on 2026-11-26. All six runs followed the frozen order and completed 40 epochs and 7,200 successful updates each. The measured accounting interval was 3,619.08 seconds, including preflight and cleanup, against the 5,400-second cap. Peak process allocated memory was 650,044,928 bytes and peak reserved memory was 824,180,736 bytes, about 0.77 GiB, against the 2 GiB reserved cap. The synthetic latency probe performed 30 forwards including warmup and zero optimizer updates. [Authorization](authorization.json) · [GPU phase ledger](gpu_phases.jsonl) · [Queue and checkpoint evidence](run_evidence.json)

The concurrent training process PID 89303 appeared in both GPU snapshots and remained active after the run. Its process and settings were left in place. The job reused the verified cache with zero external experiment downloads. Hosted CI, official test, NWPU/WiderPerson and Stage 2 remain `not_run`. [Start snapshot](background_start.txt) · [End snapshot](background_end.txt)

| Resource | CPU preparation |
| --- | --- |
| Existing local CPU environments | Windows Python 3.13 and WSL Python 3.10/3.11 with CUDA hidden |
| Real data | Existing B manifest checked against its digest, existing image paths checked, crop coordinates recomputed from manifest metadata |
| Weights | Existing `IMAGENET1K_V1` ResNet18 cache verified by SHA-256 and loaded on CPU |
| Project optimizer updates | 0 |
| GPU preflight, training, validation, profile and allocation | `not_run` |
| External dataset, checkpoint, dependency or experiment files downloaded | 0 bytes |
| Hosted CI, official test, NWPU/WiderPerson, Stage 2 | `not_run` |

Local CPU tests use generated inputs, including disposable synthetic optimizer updates for the test fixtures. The annotated source and repository metadata were consulted to establish scope and provenance. Version control publishes the implementation and compact records. The CPU phase made no changes to concurrent training. [Input and archive evidence](initialization_audit.json) · [Proposed limits](resource_plan.md)
