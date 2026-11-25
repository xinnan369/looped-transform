# LC-005 resource audit

New external experiment downloads totaled **0 bytes**. All training, validation and unit tests used existing local files and installed environments.

| Existing local input | Bytes | SHA-256 |
|---|---:|---|
| `/home/xinnan/loopcrowd-lc002/shanghaitech_a.jsonl` | 9,896,828 | `e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316` |
| `/home/xinnan/loopcrowd-lc002/torch/hub/checkpoints/resnet18-f37072fd.pth` | 46,830,571 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

Preflight hashed both files and checked the train270/internal-val30 image paths and labels. New runs started from the cached ResNet18 initialization and fresh seeded model weights. The model's pretrained path loads directly from the local cache and fails on a missing file. Mirror probes, package installation, archive extraction and external data, weight and source acquisition were skipped. NWPU and WiderPerson remained paused. The official test files were unused.

Raw histories, training logs, checkpoints and images remain local. The completed run is under `/home/xinnan/loopcrowd-lc005-run`. The initial device-index failure remains in `/home/xinnan/loopcrowd-lc005-queue.log`. The repository contains code, compact derived metrics and SHA-256 provenance. The earlier LC-002R hosted CI transfer allowance was already exhausted. LC-005 commits use `[skip ci]`. GitHub access was limited to repository coordination and publication.
