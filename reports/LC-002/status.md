# LC-002 status

LC-002 is based on LC-001 PR #2 while that prerequisite is awaiting merge. The LC-002 review branch targets the LC-001 branch.

| Task | Result | Evidence |
|---|---|---|
| T1 data protocol and capacity | ShanghaiTech A/B train and internal validation audited across three seeds and both crop policies. NWPU full-image/crop audit pending original images. | [Protocol](../../docs/DATA_AND_EVAL_PROTOCOL.md), [24 crop groups](crop_audit_summary.json) |
| T2 fixed-loop controls | CNN 0, shared 1/2/4 and independent 4 use common backbone, point head, width and loss. | [Model specification](model_specs.md) |
| T3 train and evaluation | CPU fit/save/resume and seam deduplication verified. Local nAP tests pass. NWPU official F1 is `not_ready`. | [Metric validation](metrics_validation.md), [tests](test_results.md) |
| T4 RTX 3090 | All five BF16 step profiles and real-image batch-one inference, two pilot epochs per variant, and a one-epoch P2PNet smoke. | [Experiment matrix](experiment_matrix.csv), [profile samples](profile_summary.json), [pilot epochs](pilot_history.json), [P2PNet smoke](p2pnet_smoke.json) |
| T5 evidence | Source revisions, configurations, metrics and SHA-256 references accompany external run manifests, logs and checkpoints. | [Test results](test_results.md), [next steps](next_steps.md) |

The 4-call controls both estimate 40,395,341,824 forward FLOPs per 512² image. Shared 4 has 1,596,614 parameters, and independent 4 has 2,931,206. The timing measurements show actual end-to-end variation and should be interpreted with the recorded sample counts and synchronization protocol.

The Part A pilot uses the 270-image training and 30-image internal-validation splits, BF16, ImageNet-initialized ResNet-18, seed 42, K=2 and native 512 crops. Each variant ran for two epochs, resuming from its own epoch-1 checkpoint. The common crops produced 33 overflow points out of 81,770 in epoch 1 and 34 out of 82,272 in epoch 2. The structured history contains the observed losses, metrics and denominators for both epochs. These runs are single-seed pilots. Checkpoint selection in the matrix uses internal-validation MAE.

| Variant | Training loss E1 → E2 | Internal-val MAE E1 → E2 |
|---|---:|---:|
| cnn_only | 0.5057 → 0.4594 | 337.63 → 322.63 |
| shared_1 | 0.5145 → 0.4664 | 358.23 → 416.20 |
| shared_2 | 0.5190 → 0.4742 | 365.63 → 500.53 |
| shared_4 | 0.5227 → 0.4809 | 387.83 → 374.03 |
| unshared_4 | 0.5181 → 0.4812 | 451.37 → 503.93 |

The original NWPU images remain unavailable in the checked Windows/WSL data roots. The train/val image-bound checks and real capacity audit are pending. Test labels were never used. The official ShanghaiTech test sets, formal full-train and official NWPU F1 were outside these evaluated runs. Dynamic routing and loop embeddings remain separate experiments.
