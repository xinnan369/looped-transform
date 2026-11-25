# LC-007B local verification

The three full suites ran against frozen training commit `6471f222dbe1480ff7fca1ef0291215f2498d32f`. Existing environments and dependencies were reused.

| Environment | Full suite | Local log and SHA-256 |
| --- | --- | --- |
| Ubuntu 22.04 WSL, Python 3.10.16, `mamba-ssm` | 103 passed, 287.37 s | `/home/xinnan/loopcrowd-lc007b-py310.log`, `43d52a3430f150508a22a9e6a5485affd5bc7a30a3b0655c2adc2791ef7257d2` |
| Ubuntu 22.04 WSL, Python 3.11.11, `sigma-r1-cu121` | 103 passed, 280.61 s | `/home/xinnan/loopcrowd-lc007b-py311.log`, `64d2942eedd16ace1d41230d71aff536e5fc2d12b2b66aa9ba07347f1c3ab5ca` |
| Windows, Python 3.13.2 | 103 passed, 58.26 s | `artifacts/lc007b-py313.log`, `2ffc68f74ad35f925d30b7acd25d40ffb713b40bb69c1ba156996456666e70db` |

EMA tests cover arithmetic, tied parameter identity, skipped updates, finite values, strict checkpoint names and position, and exception-safe parameter/mode/BN/RNG preservation. The training entry point produces the same live model, optimizer, schedule, loss, gradient summaries and crop sequence with EMA on or off on a synthetic dataset. Pause/resume reproduces the uninterrupted shadow and both readouts. The first development fixture used a tensor comparator on a string-valued scheduler field. Exact built-in equality checks the metadata in the frozen fixture. Existing staged/regular tiled inference and old checkpoint fixtures continue to pass.

The RTX 3090 preflight ran 48 disposable synthetic updates across eight model/seed cases, 3 each with EMA off and on. It checked exact per-step live parameter/buffer, gradient, optimizer, RNG and loss traces and the swap restoration. Those GPU phases counted toward the 5,400-second limit. The reference comparisons used generated inputs and images. [Preflight evidence](preflight.json)

The reporting fixture also checks paired endpoints, macro summaries, density alignment and the complete, stopped and local nAP-guard cases. Archival validates checkpoint and image-record SHA-256 values, restored FP32 shadow keys and step position, aligned IDs and grouped count reconciliation before recording the screen decision. An independent comparison with the corresponding LC-006 arms reproduced final live MAE, RMSE, local nAP, final training loss and crop-order hashes exactly for all 320 epochs.

An independent CSV check verified 24 indexed file digests, 640 validation sweeps, 1,296 GPU phase entries and the unrounded per-run E36–40 CV and accuracy endpoints. Running the archive command against the saved data alone regenerated 33 files byte-for-byte.

```bash
source artifacts/lc002-env.sh
python -m pytest -q --tb=short
pkgs="$HOME/anaconda3/pkgs"
PYTHONPATH="$PWD/src:$pkgs/jsonschema-4.17.3-py311h06a4308_0/lib/python3.11/site-packages:$pkgs/pyrsistent-0.18.0-py311h5eee18b_0/lib/python3.11/site-packages:$pkgs/attrs-22.1.0-py311h06a4308_0/lib/python3.11/site-packages" \
  "$HOME/anaconda3/envs/mamba-ssm/bin/python" -m pytest -q --tb=short
```

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
py -3.13 -m pytest -q --tb=short
```

Hosted CI is `not_run`. Every commit uses `[skip ci]`.
