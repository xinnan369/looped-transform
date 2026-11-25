# LC-004B resource audit

New external experiment resource downloads totaled **0 bytes**. This diagnostic ran against the following preexisting WSL files and verified their SHA-256 digests before using the frozen checkpoints.

| Existing local input | Bytes | SHA-256 |
|---|---:|---|
| `/home/xinnan/loopcrowd-lc002/shanghaitech_a.jsonl` | 9,896,828 | `e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316` |
| `/home/xinnan/loopcrowd-lc002/torch/hub/checkpoints/resnet18-f37072fd.pth` | 46,830,571 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

The only loaded model weights were the 12 existing `last.pt` files under `/home/xinnan/loopcrowd-lc003/train/warmup_cosine`. All initialization was checkpoint-based. Only the 60 selected original training images and 30 internal-validation images were opened from the fixed ShanghaiTech A manifest. The official test split was unused. NWPU and WiderPerson remain paused.

The run used installed WSL and Windows test environments. Mirror probes, package installations and external data, weights or source acquisition were skipped. Raw runtime logs, checkpoint weights and image files remain in local storage. The repository contains the prescribed fit image IDs and hashes plus derived metrics. All LC-004B commits carry `[skip ci]`. The previous hosted CI transfer allowance was exhausted before this task. GitHub access was limited to repository coordination and publication of source and derived reports.
