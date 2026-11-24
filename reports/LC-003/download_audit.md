# LC-003 resource audit

LC-003 acquired **0 bytes of external datasets, weights, dependencies or other experiment files**. The GPU queue used the following files already on WSL's local filesystem. The launch preflight verified the full SHA-256 of both files before starting CUDA work.

| Existing local input | Bytes | SHA-256 |
|---|---:|---|
| `/home/xinnan/loopcrowd-lc002/shanghaitech_a.jsonl` | 9,896,828 | `e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316` |
| `/home/xinnan/loopcrowd-lc002/torch/hub/checkpoints/resnet18-f37072fd.pth` | 46,830,571 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

Training and validation read only the local ShanghaiTech Part A files referenced by the frozen manifest under `/home/xinnan/loopcrowd-lc002/data/part_A_final`. Historical best checkpoints came from `/home/xinnan/loopcrowd-lc002r/train/42`. Diagnostics, new checkpoints and logs were written under `/home/xinnan/loopcrowd-lc003`. Cached weight loading uses a local path and fails if the file is absent. The upstream weight identifier in provenance is metadata.

Mirror probes, resource endpoint requests, downloads, pip installs and archive extraction were all skipped. NWPU and WiderPerson remained outside the experiment scope. No official test image was opened for model, threshold or schedule selection. The prior LC-002R wheel approval was exhausted and left unused for LC-003. GitHub issue and PR metadata were used for task coordination and publication.
