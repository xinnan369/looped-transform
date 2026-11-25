# LC-007B Stage 1 resource audit

This run used **0 bytes of new external experiment resources**. The program verifies the cached B manifest and ResNet-18 weight hash and uses split-scoped training/validation records. It opens images from the original development population only and uses one generated local synthetic validation fixture in preflight. No mirrors, package installs, downloads, archives, NWPU/WiderPerson or official test images were accessed in this execution.

| Existing WSL input | SHA-256 |
| --- | --- |
| `/home/xinnan/loopcrowd-lc002/shanghaitech_b.jsonl` | `de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482` |
| `/home/xinnan/loopcrowd-lc002/torch/hub/checkpoints/resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

The verified cleaned split is 360 train images / 44,856 GT heads and 40 internal validation images / 4,295 heads. GPU costs in the phase ledger include disposable synthetic preflight, initialization, successful project training, both validation readouts and checkpoint transfers. Per-process reserved memory uses the Torch CUDA allocator peak. See the [frozen preflight](preflight.json), [measured phases](gpu_phases.csv) and [queue result](research_decision.json).

Local output is `/home/xinnan/loopcrowd-lc007b`. The queue log is `/home/xinnan/loopcrowd-lc007b-queue.log`. Raw images, candidate coordinates, checkpoint tensors, optimizer state, training logs and local JSON remain on the machine. GitHub activity is limited to repository coordination and code/evidence publication. Previous hosted CI transfer authorization is exhausted and all commits carry `[skip ci]`.

| Local execution evidence | SHA-256 |
| --- | --- |
| `/home/xinnan/loopcrowd-lc007b-queue.log` | `fd32273f35b20fe066b7590105649e99addbc36288a812ae5b739610af5541ed` |
| `/home/xinnan/loopcrowd-lc007b/gpu_phases.jsonl` | `8557eecf0788771d229fba35ecbb092df031ad4a2e5de1c7afb883367e4381b0` |
| `/home/xinnan/loopcrowd-lc007b/queue_progress.json` | `f5025987a76da5506f1125da03302a4bd0350c1652c7ca329ab51875484bff40` |

No route or redirect probe was used. Future file acquisition requires a new approval with reliable mainland sources and final endpoints, planned byte totals, checksums, cache and disk capacity, and the applicable route verification. Stage 1 authorizes no downloads for any scope.
