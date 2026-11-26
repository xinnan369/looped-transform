# Executed resource audit

| Resource | Current phase |
| --- | --- |
| Existing local CPU environments | Windows Python 3.13 and WSL Python 3.10/3.11 with CUDA hidden |
| Real data | Existing B manifest checked against its digest, existing image paths checked, crop coordinates recomputed from manifest metadata |
| Weights | Existing `IMAGENET1K_V1` ResNet18 cache verified by SHA-256 and loaded on CPU |
| Project optimizer updates | 0 |
| GPU preflight, training, validation, profile and allocation | `not_run` |
| External dataset, checkpoint, dependency or experiment files downloaded | 0 bytes |
| Hosted CI, official test, NWPU/WiderPerson, Stage 2 | `not_run` |

Local CPU tests use generated inputs, including disposable synthetic optimizer updates for the test fixtures. The annotated source and repository metadata were consulted to establish scope and provenance. Version control publishes the implementation and compact records. The CPU phase made no changes to concurrent training. [Input and archive evidence](initialization_audit.json) · [Proposed limits](resource_plan.md)
