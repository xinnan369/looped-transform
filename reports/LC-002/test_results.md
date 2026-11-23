# LC-002 verification

Ubuntu 22.04 on WSL2 ran Python 3.11.11, PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, NumPy 1.26.4 and SciPy 1.13.0. The GPU was an NVIDIA GeForce RTX 3090, 24 GiB. CUDA 12.1 reported BF16 support and a FP16 allocation/reduction returned 1.0.

| Check | Observed result |
|---|---|
| WSL2 `pytest -q` | **48 passed in 19.91 s** |
| CLI startup | `train.py`, `evaluate.py`, `profile.py` and `audit_crops.py --help` all exited 0 |
| CPU fixture | Fit, best/last checkpoint, resume to epoch 2, reload, full-image evaluation and official-test guard passed |
| Sharing | All four invocations observed the same attention parameter and returned separate nonzero direct gradients. Four distinct parameters were observed for the independent control. |
| CUDA | All 5 variants completed forward/backward/AdamW, batch 2 at 512² BF16. All ran evaluation with real-image batch-one tiles. See [profile data](profile_summary.json). |
| Real checkpoint reload | `shared_4` best checkpoint reproduced all saved internal-validation count and local nAP values. [Reload evidence](checkpoint_eval.json) |
| P2PNet upstream | One training-only epoch completed on CUDA using the cleaned ShanghaiTech A train list. Upstream skips validation at epoch index 0. [Smoke evidence](p2pnet_smoke.json) |
| Precision validation | P2PNet-style local nAP and count metrics pass numeric toy cases. See [metric checks](metrics_validation.md). |
| Known data blocker | Original NWPU images were absent from the checked roots. NWPU image-bound capacity and official F1 remain pending. |

## Reproduction

From the repository root, use the existing WSL environment or install matching Torch/CUDA dependencies separately before `python -m pip install -e '.[test,torch]'`. On hosted CPU CI the workflow installs torch and torchvision from the CPU wheel index. Set `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` for the local commands below.

```sh
pytest -q
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/profile.py --help
python scripts/audit_crops.py --help
```

The original ShanghaiTech A/B archive and point rejection policy are documented in the [LC-001 report](../LC-001/status.md). Both derived manifests were rebuilt on WSL. SHA-256 values were `e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316` (A) and `de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482` (B). Path strings are part of the manifest hashes. The full 24-group per-image crop audit is at `~/loopcrowd-lc002/crop_audit_per_image.json` with SHA-256 `0a03c1161978f09e4348a7f035dc09308fc003609193b6b9c1070f5b4b2a75c5`.

```sh
work="$HOME/loopcrowd-lc002"
export TORCH_HOME="$work/torch"
python scripts/audit_crops.py --manifest "$work/shanghaitech_a.jsonl" --manifest "$work/shanghaitech_b.jsonl" --output "$work/crop_audit_per_image.json"
for variant in cnn_only shared_1 shared_2 shared_4 unshared_4; do
  python scripts/profile.py --config configs/lc002_profile.json --variant "$variant" --device cuda --manifest "$work/shanghaitech_a.jsonl" --output "$work/profile/$variant"
  python scripts/train.py --config configs/lc002_pilot.json --variant "$variant" --device cuda --manifest "$work/shanghaitech_a.jsonl" --output "$work/pilot/$variant"
done
```

To reproduce the pilot continuation, set `epochs` to 2 in a copy of `configs/lc002_pilot.json` and repeat the same training command with that config and `--resume` for each output directory. The local run uses `~/loopcrowd-lc002/pilot_two_epochs.json`. Each variant's first-epoch manifest, log and last checkpoint remain archived under its run directory. To repeat validation from disk, run the following using the saved checkpoint.

```sh
python scripts/evaluate.py --checkpoint "$work/pilot/shared_4/best.pt" --manifest "$work/shanghaitech_a.jsonl" --device cuda --output "$work/pilot/shared_4-final-eval"
```

Each profile/pilot directory has a `run_manifest.json` with the command, status, git SHA, data SHA, config, label policy and environment. Training manifests record best/last checkpoint SHA-256. Full logs and weights remain under `~/loopcrowd-lc002`. The official ResNet-18 checkpoint was acquired after initial CDN timeout by assembling byte ranges and verifying SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` before loading.

## P2PNet upstream smoke

The upstream commit and torchvision version-check correction are in the [baseline guide](../../docs/BASELINE_P2PNET.md). The first attempt stopped before training because `models/vgg_.py` points to the authors' private filesystem. In the separate upstream checkout, replace `/apdcephfs/private_changanwang/checkpoints/vgg16_bn-6c64b313.pth` with the absolute path to the local `$TORCH_HOME/hub/checkpoints/vgg16_bn-6c64b313.pth`. The verified weight SHA-256 was `6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a`. The rerun completed epoch index 0 in 6.02 s according to upstream. Its exact training command, log and artifact hashes are in [p2pnet_smoke.json](p2pnet_smoke.json). P2PNet validation metrics are `not_run`.
