# P2PNet baseline

Status for every accuracy metric is `not_run`. LC-001 prepares the baseline inputs, commands and evaluation contract. Train only in a subsequent approved run.

Use the Tencent Youtu [official repository](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/tree/5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455) at commit `5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455` and the [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Song_Rethinking_Counting_and_Localization_in_Crowds_A_Purely_Point-Based_Framework_ICCV_2021_paper.html). Its [license](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455/LICENSE) limits P2PNet use to academic research and requires retaining the applicable notices for redistribution. Keep the upstream checkout, pretrained weights and datasets outside the LoopCrowd Git tree. The [NWPU release terms](https://huggingface.co/datasets/Junyu3/NWPU-Crowd/blob/main/readme.md) additionally restrict data redistribution.

## Environment and input preparation

Upstream reports Python 3.6.5 and PyTorch 1.5.0. For RTX 3090, use a Linux/WSL environment with an Ampere-capable CUDA PyTorch build. LC-001 checked CUDA in WSL with Python 3.11, PyTorch 2.2.2 and CUDA 12.1. The pinned upstream code needs a version-check correction because its float conversion reads torchvision 0.17 as 0.1 and imports APIs removed after 0.6. Apply this correction in the upstream checkout and save the diff with each run.

```sh
git clone https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet.git
cd CrowdCounting-P2PNet
git checkout --detach 5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'numpy<2' 'opencv-python<4.12' -r requirements.txt
python - <<'PY'
from pathlib import Path
path = Path("util/misc.py")
text = path.read_text()
old = "float(torchvision.__version__[:3]) < 0.7"
new = "tuple(map(int, torchvision.__version__.split('.')[:2])) < (0, 7)"
assert text.count(old) == 2
path.write_text(text.replace(old, new))
PY
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda').sum().item())"
python train.py --help
mkdir -p /path/to/experiment
python -m pip freeze > /path/to/experiment/p2pnet-environment.txt
git diff > /path/to/experiment/p2pnet-compatibility.diff
```

From LoopCrowd in its own Linux/WSL environment, build the ShanghaiTech manifest as described in the README and prepare one derived directory per part. The option `--evaluation-split val` uses only the seed-42 internal validation for checkpoint selection. The upstream loader calls that list `shanghai_tech_part_a_test.list`; the adapter writes the validation members into it for this run. Use `--evaluation-split test` and a second directory after model selection. The adapter writes one `x y` pair per line, and links the original images. Part B uses the same list format and `SHHA` loader in a separate run.

```sh
python scripts/prepare_p2pnet.py --manifest artifacts/shanghaitech_a.jsonl --output "$PWD/artifacts/p2pnet-a-val" --evaluation-split val
python scripts/prepare_p2pnet.py --manifest artifacts/shanghaitech_a.jsonl --output "$PWD/artifacts/p2pnet-a-test" --evaluation-split test
```

With the prepared validation root, run these commands from the pinned upstream checkout. The VGG16 backbone requires its ImageNet weights in the Torch cache or download access. Keep the upstream ImageNet normalization, random scaling, 128 × 128 crop and horizontal-flip settings for this baseline. LoopCrowd's 512 × 512 candidate audit has separate settings.

```sh
export DATA_ROOT=/absolute/path/to/loopcrowd/artifacts/p2pnet-a-val
export RUN_ROOT=/absolute/path/to/experiment
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/weights" "$RUN_ROOT/tensorboard"
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root "$DATA_ROOT" --dataset_file SHHA --epochs 3500 --lr_drop 3500 \
  --output_dir "$RUN_ROOT/logs" --checkpoints_dir "$RUN_ROOT/weights" \
  --tensorboard_dir "$RUN_ROOT/tensorboard" --lr 0.0001 --lr_backbone 0.00001 \
  --batch_size 8 --eval_freq 1 --gpu_id 0 --seed 42 2>&1 | tee "$RUN_ROOT/train.log"
```

The released `build_dataset` dispatcher registers only `SHHA`. A separate NWPU adapter needs to preserve official train/val membership, normalize the empty-point shape to `(0, 2)` and add image-only test inference. Zero-head images are part of the dataset. The real NWPU images and this upstream integration are pending.

## Counting evaluation

Use the frozen checkpoint and the dedicated test data root. In the upstream code, `evaluate_crowd_no_overlap` counts foreground scores strictly greater than 0.5. Across images it computes MAE as the mean absolute count difference, and RMSE as the square root of the mean squared count difference. The upstream log labels the latter `mse`. The upstream `run_test.py` only produces demo visualizations. To invoke the underlying counting evaluator directly, run this from the pinned upstream checkout.

```sh
export EVAL_ROOT=/absolute/path/to/loopcrowd/artifacts/p2pnet-a-test
export P2PNET_CHECKPOINT="$RUN_ROOT/weights/best_mae.pth"
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import argparse
import os
import torch
from crowd_datasets import build_dataset
from engine import evaluate_crowd_no_overlap
from models import build_model
from train import get_args_parser
from util.misc import collate_fn_crowd

parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args(["--dataset_file", "SHHA", "--data_root", os.environ["EVAL_ROOT"]])
device = torch.device("cuda")
model = build_model(args, training=False).to(device)
checkpoint = torch.load(os.environ["P2PNET_CHECKPOINT"], map_location="cpu")
model.load_state_dict(checkpoint["model"])
_, dataset = build_dataset(args)(args.data_root)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn_crowd)
mae, rmse = evaluate_crowd_no_overlap(model, loader, device)
print({"mae": float(mae), "rmse": float(rmse)})
PY
```

If using the LC-001 rejection option, evaluation uses the cleaned labels. Record that protocol explicitly. For comparison on original ShanghaiTech counts, prepare a second evaluation list with the unmodified official annotations converted to the upstream text format and keep that label policy in the run record. Keep one report for each policy.

## Localization evaluation

- **P2PNet paper nAP.** Export each prediction's image ID, native-resolution `x,y` and foreground score before score thresholding. Under paper §3.2, rank predictions by descending score, match each to at most one as-yet-unmatched ground-truth point within the same image, and count unmatched predictions as false positives. A candidate match requires the Euclidean distance divided by the ground-truth point's mean distance to its `k` nearest neighbors to be strictly below `δ`. Compute AP on the precision–recall curve. The paper uses `k=3` in its protocol illustration and reports nAP across `δ=0.05,0.10,…,0.50`, including the mean across these thresholds. The pinned public P2PNet checkout does not contain a standalone nAP evaluator. A reproduction must state the sparse/duplicate-point handling, tie handling and AP integration convention and validate the implementation before reporting comparable nAP. [Paper and supplements](https://openaccess.thecvf.com/content/ICCV2021/html/Song_Rethinking_Counting_and_Localization_in_Crowds_A_Purely_Point-Based_Framework_ICCV_2021_paper.html)
- **NWPU benchmark localization.** Report micro precision, recall and F1 using one-to-one assignment under the benchmark's small and large per-head distance tolerances derived from head boxes. Use the authors' validation code and original benchmark ground truth. At the pinned evaluator revision, set `pred_file` in `eval/eval.py` to your exported validation localization file and run `cd eval && python eval.py`. For the official test benchmark, submit one line per test image, `image_id N x1 y1 … xN yN`, with integer coordinates in original image space. For official test counting, submit one line per test image in the separate `image_id predicted_count` format. The hidden NWPU test ground truth is evaluated by the official server. Record the frozen confidence threshold and coordinate rounding. [Validation evaluator](https://github.com/gjy3035/NWPU-Crowd-Sample-Code-for-Localization/blob/72ccfa28d481af890bf51c88afdef02e67a33577/eval/eval.py) · [Benchmark submission](https://www.nwpu-optimal.cn/nwpucrowdloc.html)
