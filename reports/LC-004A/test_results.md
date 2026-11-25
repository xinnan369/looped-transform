# LC-004A verification and reproduction

The training and test code is at `12dea46e707ad38d854835351b1e8d3e03f58cac`. All tests used preinstalled interpreters and existing local dependencies.

| Local interpreter | Command | Result |
|---|---|---|
| WSL Ubuntu 22.04, Python 3.11.11, torch 2.2.2+cu121, torchvision 0.17.2+cu121 | `pytest -q` | 73 passed in 39.53 s |
| WSL Ubuntu 22.04, Python 3.10.16, torch 2.3.1+cu118, torchvision 0.18.1+cu118 | `python -m pytest -q` with cached pure-Python schema packages | 73 passed in 44.55 s |
| Windows, Python 3.13.2, torch 2.11.0, torchvision 0.26.0 | `py -3.13 -m pytest -q` | 73 passed in 73.34 s with AnyIO and GradScaler compatibility deprecation warnings |

The targeted WSL run passed 37 tests in 46.08 s. New tests cover unchanged BN buffers under `train()`, nonzero affine gradients, old configuration defaults, exact g=1 outputs and core gradients, shared parameter identity across actual calls, independent stage entries, fixed-prefix traces, posterior removal of the second call and competition between neighboring candidate matches. Every historical full-depth checkpoint reproduced the archived whole-split count and local nAP. Each historical Shared-2 all-tiles two-call trace also produced bitwise identical decoded arrays to the ordinary inference path.

## Local test commands

```bash
source artifacts/lc002-env.sh
pytest -q

pkgs="$HOME/anaconda3/pkgs"
PYTHONPATH="$PWD/src:$pkgs/jsonschema-4.17.3-py311h06a4308_0/lib/python3.11/site-packages:$pkgs/pyrsistent-0.18.0-py311h5eee18b_0/lib/python3.11/site-packages:$pkgs/attrs-22.1.0-py311h06a4308_0/lib/python3.11/site-packages" \
  "$HOME/anaconda3/envs/mamba-ssm/bin/python" -m pytest -q
```

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
py -3.13 -m pytest -q
```

## CUDA queue

```bash
source "$HOME/.venvs/sigma-r1-cu121/bin/activate"
export PYTHONPATH="$PWD/src:$HOME/loopcrowd-lc002/deps"
export TORCH_HOME="$HOME/loopcrowd-lc002/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m scripts.run_lc004a \
  --manifest "$HOME/loopcrowd-lc002/shanghaitech_a.jsonl" \
  --lc003 "$HOME/loopcrowd-lc003" \
  --output "$HOME/loopcrowd-lc004a-reproduction"
```

All model measurements ran serially on one local RTX 3090 with the WSL Python 3.11 environment and CUDA 12.1 runtime. The sample timing protocol uses one warm-up step and nine synchronized post-warm-up repetitions, synthetic batch-two resident 512 px training inputs and the same full validation image. [Run commands, times and hashes](training_history.json)

A timing recheck under concurrent workstation load used the local command `python artifacts/lc004a-interleaved-profile.py` in the same WSL environment. It rotates the three matched variants over nine trials with synchronization before and after each measurement. The index records the recheck script hash, raw samples, full command and GPU telemetry.

Hosted CI is outside the authorized resource scope.
