# LC-006 resource audit

New external experiment downloads totaled **0 bytes**. The before-training audit checked the following existing local inputs.

| WSL file | Bytes | SHA-256 |
|---|---:|---|
| `/home/xinnan/loopcrowd-lc002/shanghaitech_b.jsonl` | 3,740,682 | `de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482` |
| `/home/xinnan/loopcrowd-lc002/torch/hub/checkpoints/resnet18-f37072fd.pth` | 46,830,571 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

All 400 original training-image and MAT pairs under `/home/xinnan/loopcrowd-lc002/data/part_B_final/train_data` decoded and matched the original train360/internal-val40 split and the historical rejection ledger. [Dataset evidence](data_audit.json)

The official test image files were excluded. NWPU and WiderPerson remained outside the experiment scope. Package installation, mirror probes, archive extraction and external file acquisition were skipped. B runs initialized from existing cached ImageNet1K_V1 weights. No A model checkpoint was used to initialize a B arm. The local pretrained path fails on a missing file.

Model weights, images, full histories and logs stay local under `/home/xinnan/loopcrowd-lc006` or the original cached paths. Git stores source and small derived summaries. All LC-006 commits use `[skip ci]`. The previous LC-002R hosted CI allowance was exhausted before this task. GitHub activity is limited to repository coordination and source/report publication.
