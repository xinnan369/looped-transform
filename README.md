# LoopCrowd

LoopCrowd studies depth-wise weight sharing in Transformers for head-point localization and counting on ShanghaiTech Parts A/B and NWPU-Crowd. The target is one RTX 3090 (24 GB), with 512 × 512 training crops, stride-4 output and future looped window attention at stride 8. LC-001 provides Stage 0 dataset manifests, candidate-capacity auditing, CPU tests and a P2PNet reproduction plan.

## Setup

Use Python 3.10 or newer. The manifest and audit tools run on CPU on Linux, WSL and Windows.

```sh
python -m venv .venv
source .venv/bin/activate        # Linux/WSL
python -m pip install -e '.[test]'
pytest -q
python scripts/audit_dataset.py --help
```

In Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. If using an existing WSL PyTorch environment, activate that environment and install `.[test]` in it. For a fresh CUDA environment, choose a driver-compatible CUDA build using the [PyTorch installer](https://pytorch.org/get-started/locally/) and run this check after installation.

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda').sum().item())"
```

Stage 0 tooling has no GPU dependency. The optional `torch` extra provides PyTorch for research environments. See the [P2PNet guide](docs/BASELINE_P2PNET.md) for upstream-specific dependencies.

## Build manifests

Get the [ShanghaiTech release](https://github.com/desenzhou/ShanghaiTechDataset) and [official NWPU-Crowd release](https://huggingface.co/datasets/Junyu3/NWPU-Crowd). Keep the original files outside Git. Run from the repository root.

```sh
python scripts/build_manifest.py --config configs/shanghaitech_a.json --root data/part_A_final --output artifacts/shanghaitech_a.jsonl
python scripts/build_manifest.py --config configs/shanghaitech_b.json --root data/part_B_final --output artifacts/shanghaitech_b.jsonl
python scripts/build_manifest.py --config configs/nwpu_crowd.json --root data/NWPU-Crowd --output artifacts/nwpu_crowd.jsonl
```

ShanghaiTech uses `train_data` and `test_data`, each with `images` and `ground_truth`. The included config assigns `floor(0.10 × train size)` images to internal validation using a SHA-256 rank of `42:<image_id>`. Official test images remain exclusively in test. NWPU uses `images`, `jsons` and the official `train.txt`, `val.txt` and `test.txt` membership. The adapter expects per-image JSON with `img_id`, `human_num` and `points`, supports the released three-column test list and the documented one-column form, and never infers split from the ID range. Set `image_suffix` explicitly if your image directory uses `.jpeg`.

Each UTF-8 JSONL record contains `dataset`, `image_id`, `split`, `image_path`, `image_width`, `image_height`, and `points`. Points are floating-point `[x,y]` in the half-open image rectangle `[0,width) × [0,height)`. `coordinate_base` explicitly defines the input origin. The manifest keeps the original image path as an absolute reference, so rebuild it when switching between Windows and WSL or moving the dataset. `[]` marks a labeled zero-head image and `null` marks an unlabeled NWPU test image. The converter rejects missing pairs, duplicate dataset/image IDs, count mismatches, and invalid coordinates. IDs and points have stable ordering. Write errors preserve a previously completed manifest.

The checked ShanghaiTech release has finite points outside the image rectangle. The strict conversion above detects them. To reproduce LC-001's cleaned manifests, add `--out-of-bounds-rejections artifacts/<dataset>_rejections.json`. This explicitly excludes finite out-of-image points and records each original point index and its coordinates after origin conversion. Other validation errors still abort. Keep the rejection log local and record the chosen policy with any downstream result.

## Audit capacity

```sh
python scripts/audit_dataset.py --manifest artifacts/shanghaitech_a.jsonl --split train --source real --output-prefix artifacts/shanghaitech_a_train
```

The CLI writes JSON and Markdown. The default runs full-image and 512-crop audits. Use `--mode` to select one mode or `--aggregate-only` to omit per-image identifiers. Reports group by dataset, split and mode, and include a pooled `all` summary when multiple labeled splits are selected. Use training-only numbers for design and validation for model selection. Unlabeled images are counted separately and excluded from point metrics.

The crop audit uses one deterministic native-resolution crop per image, with seed 42 by default. On each axis the SHA-256 digest of `seed:dataset:image_id` picks an integer origin in the available range. An axis shorter than 512 has origin 0 and is padded on the right or bottom. Coordinates are translated into the crop before grid assignment. Cropped-away points are tracked through the difference between source and audited head counts.

At stride `s`, each point belongs to cell `(floor(x/s), floor(y/s))`. The total number of cells is `ceil(width/s) × ceil(height/s)`, including empty and partial edge cells. For a cell with `n` points, capacity `K` drops `max(0,n-K)` points. Point loss rate is total dropped divided by total audited points, with `null` for a zero denominator. The pooled rate uses pooled counts. Nearest-neighbor percentiles pool Euclidean distances computed within each image or crop, one distance per point with a neighbor. Images with fewer than two points contribute no distances. Coincident points retain distance zero.

See the [workflow](docs/WORKFLOW.md), [status](reports/LC-001/status.md) and [report schema](reports/LC-001/data_audit.schema.json).
