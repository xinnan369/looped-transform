# LC-005 verification and reproduction

Tests and training used the existing local environments. Training code ran at `a0980795e425d8ccbaae0d400ace31efa06b1687`.

| Interpreter | Command | Result |
|---|---|---|
| WSL Ubuntu 22.04, Python 3.11.11, torch 2.2.2+cu121, torchvision 0.17.2+cu121 | `pytest -q` | 84 passed in 116.23 s |
| WSL Ubuntu 22.04, Python 3.10.16, torch 2.3.1+cu118, torchvision 0.18.1+cu118 | `python -m pytest -q` with existing pure-Python schema packages | 84 passed in 117.51 s |
| Windows, Python 3.13.2, torch 2.11.0, torchvision 0.26.0 | `py -3.13 -m pytest -q` | 84 passed in 58.58 s, with three compatibility deprecation warnings from AnyIO and GradScaler |

The first focused run passed seven tests in 201.29 s. Its new fixtures check two calls through the identical block and decoder objects, 1,596,614 parameters, stage shape and finite gradients, nonzero auxiliary contributions to the shared block and both heads, legacy checkpoint loading and bitwise agreement of ordinary and staged final inference. The λ=0 stage loss is bitwise identical to the legacy loss in the CPU fixture, and the core gradient comparison uses relative tolerance 1e−6 and absolute tolerance 1e−7. A synthetic CPU fixture completes one optimizer update, saves and resumes at the same horizon, and verifies staged loss, match counts, initialization and crop hashes.

The first queue attempt stopped during device memory inspection before training. The installed PyTorch requires an explicit index for `mem_get_info`. The compatibility commit adds `cuda:0` for preflight. The successful queue started in a fresh output directory with the frozen protocol and configurations. Both raw queue logs are identified in [run evidence](run_evidence.json).

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

## Audited RTX 3090 queue

```bash
source "$HOME/.venvs/sigma-r1-cu121/bin/activate"
export PYTHONPATH="$PWD/src:$HOME/loopcrowd-lc002/deps"
export TORCH_HOME="$HOME/loopcrowd-lc002/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m scripts.run_lc005 \
  --manifest "$HOME/loopcrowd-lc002/shanghaitech_a.jsonl" \
  --output "$HOME/loopcrowd-lc005-reproduction"
```

The queue writes its fairness and warm-up checks before starting the six ordered training arms. Hosted CI is outside the authorized resource scope.
