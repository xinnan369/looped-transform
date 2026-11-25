# LC-007A offline experiment resource audit

New external experiment files, package installs, pretrained weight retrievals, mirror or route probes, archive extraction, official test image opens and project optimizer updates total **zero**. The replay loads only local archived E40 weights into models built with `initialize=False`. It lists train and internal-val image files from the two frozen manifests, then opens development-val only. The historical A label policy removed 294 finite out-of-image points from the original 300 development images. The B development replay logged 118 train and 11 val rejections. LC-007A retained the manifest membership and labels from those archives. [A data policy](../LC-001/status.md) · [B data audit](../LC-006/data_audit.json)

| Existing WSL input | Verified SHA-256 |
| --- | --- |
| `/home/xinnan/loopcrowd-lc002/shanghaitech_a.jsonl` | `e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316` |
| `/home/xinnan/loopcrowd-lc002/shanghaitech_b.jsonl` | `de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482` |
| 33 locally cached E40 checkpoints | [Frozen digest and path list](frozen_cases.csv) |
| Historical local inputs referenced by the ledger | [Best/last, raw history and manifest digest checks](run_ledger.csv) |

NWPU/WiderPerson access, preparation and evaluation remain paused. Download origins, final redirection, route, archive CRC and resume checks are inapplicable to this zero-download experiment. An early module-path import error and a pre-inference ledger classification assertion both stopped before validation image access. The completed replay output remains at `/home/xinnan/loopcrowd-lc007a`.

| WSL log | SHA-256 |
| --- | --- |
| `/home/xinnan/loopcrowd-lc007a-preflight-01.log` | `aa16ff7ead2db8da26826331c899b2ceadb62c5fbe0a2a4fa6aa0c3ff508b7f2` |
| `/home/xinnan/loopcrowd-lc007a-inference.log` | `ab107952ac62b1be6719088ec045e5cdc926e5acb5fb968886947bec543a88ba` |

The previous LC-002R hosted CI authorization is exhausted. All LC-007A commits carry `[skip ci]`. Only GitHub reads for coordination and publication of code and derived reports use repository network access. Local datasets, weights, full epoch logs and candidate coordinates remain local. Any future resource request requires separate authorization with exact bytes, local cache and space, trusted mainland source and final redirect, hashes and permitted route checks before acquisition.
