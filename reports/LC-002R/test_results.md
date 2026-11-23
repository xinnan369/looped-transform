# Tests and reproduction

The GPU runs use Ubuntu 22.04 under WSL2, Python 3.11.11, torch 2.2.2+cu121, torchvision 0.17.2+cu121 and **NVIDIA GeForce RTX 3090 24 GiB**. The training checkout is identified in each run by its actual git SHA. CUDA profiles use BF16 with nine post-warm-up timings per measurement and report allocated and reserved memory.

| Check | Observed result |
|---|---|
| WSL CPU `pytest -q` | 55 passed in 17.01 s before the CUDA queue and again in 23.03 s after adding the frozen-test sweep rejection assertion |
| Windows CPU `pytest -q` | 55 passed in 46.50 s with one third-party AnyIO deprecation warning |
| Close points | In-tile and seam-straddling true neighbors 1 px apart survive AP and counting |
| Exact overlap duplicate | One owning tile contributes to counting. Both predictions enter AP and the duplicate is unmatched. |
| Local metric | Eight fixtures verified global rank, GT associations and AP against a dense independent oracle at each δ. Maximum absolute AP difference 0.0. |
| Shifted window | Direct block perturbation crosses a former boundary on the shifted call and produces no opposite-edge influence. Padding mask, shared object identity, alternating shifts and gradients passed. |
| CPU end to end | Checkpoint save, resume and full-image evaluation, including 19 internal validation thresholds, passed on the fixture. |

## Reproduce the training and evidence

Use the LC-002 WSL environment and data manifest `~/loopcrowd-lc002/shanghaitech_a.jsonl`, SHA-256 `e026bb299a54f28b85c245e707f952361684b5bb0db9e50f031679950f641316`. Use the official torchvision ResNet-18 ImageNet weights SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and `TORCH_HOME=~/loopcrowd-lc002/torch`. For a comparable rerun, check out the code revision stored per run and set `PYTHONPATH` to `src` or use an editable install. Keep new outputs in separate directories.

```bash
pytest -q
python scripts/verify_nap.py --upstream-archive artifacts/p2pnet-upstream.tar.gz --output /tmp/lc002r-metric-parity.json
work="$HOME/loopcrowd-lc002r"
manifest="$HOME/loopcrowd-lc002/shanghaitech_a.jsonl"
for variant in cnn_only shared_1 shared_2 shared_4 unshared_4 shifted_shared_2 shifted_shared_4; do
  python scripts/profile.py --config configs/lc002_profile.json --variant "$variant" --device cuda --manifest "$manifest" --repeats 9 --output "$work/profile/$variant"
done
for variant in cnn_only shared_1 shared_2 shared_4 unshared_4 shifted_shared_2 shifted_shared_4; do
  python scripts/train.py --config configs/lc002r_20epoch.json --variant "$variant" --device cuda --manifest "$manifest" --output "$work/train/42/$variant"
  python scripts/evaluate.py --checkpoint "$work/train/42/$variant/best.pt" --manifest "$manifest" --device cuda --threshold-sweep --output "$work/sweep/42/$variant"
done
```

The initial fixture-check CLI imported training provenance helpers and encountered the local `scripts/profile.py` versus standard-library `profile` name collision. The verification CLI now uses only the numeric metric, JSON and hashlib and completed successfully. The sample dataset, predictions, checkpoints and full process logs stay outside Git. The path and digest of each completed run's manifest, log and best/last checkpoints are in [training_history.json](training_history.json). The canonical resolved config hash uses sorted JSON keys and compact separators.
