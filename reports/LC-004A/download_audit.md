# LC-004A resource audit

New external experiment resources downloaded **0 bytes**. LC-004A reads the following preexisting files from WSL's local filesystem. The queue verified both SHA-256 digests before evaluation or training.

| Existing local input | Bytes | SHA-256 |
|---|---:|---|
| `/home/xinnan/loopcrowd-lc002/shanghaitech_a.jsonl` | 9,896,828 | `e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316` |
| `/home/xinnan/loopcrowd-lc002/torch/hub/checkpoints/resnet18-f37072fd.pth` | 46,830,571 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

Images were read from the frozen ShanghaiTech Part A train and internal-validation entries under `/home/xinnan/loopcrowd-lc002/data/part_A_final`. The 24 historical snapshots and their training histories and logs were read from `/home/xinnan/loopcrowd-lc003/train/warmup_cosine`. The queue verified their archived checkpoints and training histories by SHA-256. New checkpoints, detailed raw diagnostics and logs remain under `/home/xinnan/loopcrowd-lc004a` and its sibling log files.

Pretrained model initialization reads the local cache directly and fails if the required file is absent. Mirror and endpoint probes, fallback sources, pip installs and archive extraction were skipped. NWPU and WiderPerson remained paused. The official test split was unused. The earlier LC-002R hosted CI allowance was exhausted before this work. LC-004A commits use `[skip ci]` and GitHub activity is limited to repository coordination and publishing source and summary reports.
