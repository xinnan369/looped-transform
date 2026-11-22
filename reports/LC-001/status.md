# LC-001 Stage 0 status

| Issue deliverable | Status | Evidence |
|---|---|---|
| Repository and experiment workflow | Complete | [README](../../README.md), [workflow](../../docs/WORKFLOW.md), [Python package](../../pyproject.toml) |
| ShanghaiTech A/B and NWPU parsing | Implemented | Real ShanghaiTech A/B manifests, all released NWPU label JSONs checked against the parser, and end-to-end synthetic tests for both layouts |
| Candidate capacity audit | Complete for ShanghaiTech A/B | [Part A audit](shanghaitech_a_audit.md), [Part B audit](shanghaitech_b_audit.md), [JSON Schema](data_audit.schema.json) |
| P2PNet preparation | Prepared | [Pinned upstream, data adapter, training commands and evaluation protocols](../../docs/BASELINE_P2PNET.md) |
| CPU tests and GitHub Actions | Implemented | [Test results](test_results.md), [workflow](../../.github/workflows/cpu-tests.yml) |
| Follow-up | Recorded | [Next steps](next_steps.md) |

## Annotation reconciliation

The [ShanghaiTech author release](https://github.com/desenzhou/ShanghaiTechDataset) archive had SHA-256 `95b7a7f3d927f756393005b98f77847ce40dcb8546e2f46d20f0e0d425dac403`. Raw counts were read from the nested `location` arrays and checked against the corresponding `number` fields. The explicit rejection option excluded finite out-of-image points under the configured zero-based image bounds. Strict mode reported invalid points in both parts. Each accepted coordinate was preserved. The aggregate files contain no image IDs, source paths or point coordinates.

| Part | Official split | Images | Raw points | Excluded | Manifest points |
|---|---|---:|---:|---:|---:|
| A | Train | 300 | 162,707 | 294 | 162,413 |
| A | Test | 182 | 78,970 | 108 | 78,862 |
| B | Train | 400 | 49,280 | 129 | 49,151 |
| B | Test | 316 | 39,208 | 87 | 39,121 |
| **Total** | | **1,198** | **330,165** | **618** | **329,547** |

Seed 42 assigns 30 of the 300 Part A training images and 40 of the 400 Part B training images to internal validation. Official test membership is preserved. Both reports use crop seed 42 and pool full-image and crop-512 statistics separately. Their `source_head_count` refers to validated manifest points before cropping.

## Pending work

- NWPU-Crowd's official JSON archive and split lists were available. Complete original images were absent from the checked Windows and WSL dataset roots, so image-bound checks, the real manifest and the real capacity audit remain pending. The official test split has no released ground truth.
- P2PNet training and metrics are `not_run`. WSL reached the RTX 3090 and passed a CUDA tensor smoke test. NWPU integration also requires extending the upstream dataset loader. The baseline guide records the remaining evaluation work, including implementation validation for paper nAP.
