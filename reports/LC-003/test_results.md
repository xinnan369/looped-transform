# LC-003 tests and reproduction

All three local interpreters passed the full 66-test suite with the training code at `34e310976d1bdbd0eab7855eb02bf087cba86638`. No dependency installation was needed.

| Interpreter | Command | Result |
|---|---|---|
| WSL Ubuntu 22.04, Python 3.11.11, torch 2.2.2+cu121, torchvision 0.17.2+cu121 | `pytest -q` | 66 passed in 169.64 s |
| Windows, Python 3.13.2, torch 2.11.0, torchvision 0.26.0 | `py -3.13 -m pytest -q` | 66 passed in 282.20 s, two deprecation warnings from AnyIO and the compatibility GradScaler API |
| WSL Ubuntu 22.04, Python 3.10.16, torch 2.3.1+cu118, torchvision 0.18.1+cu118 | `python -m pytest -q` with local pure-Python schema packages below | 66 passed in 24.13 s |

The initial Python 3.10 collection used the LC-002 Python 3.11 dependency directory and failed to import its ABI-specific `rpds.rpds` extension. Running with that directory removed passed 59 non-audit tests. Then the full suite ran using the preexisting conda package cache's jsonschema 4.17.3, pyrsistent 0.18.0 and attrs 22.1.0 on `PYTHONPATH`. The cached pyrsistent package used its pure-Python `_pvector` implementation.

The scheduler tests check the 270 warm-up steps and both endpoints of the 5,400-update trajectory. The interrupted CPU fixture resumes at the same update with identical LR history, final model tensors and optimizer state. It verifies that the disabled AMP state survives serialization and rejects a changed horizon. The previous constant config and extend-epochs smoke test remain functional. The matched controls test identical first-block and head initialization, distinct parameters for independent blocks, expected shifts and finite nonzero gradients. A separate fixture exercises missing local pretrained weights.

## Local test commands

```bash
# WSL, from the repository root. The env file is a local LC-002 work aid.
source artifacts/lc002-env.sh
pytest -q

pkgs="$HOME/anaconda3/pkgs"
PYTHONPATH="$PWD/src:$pkgs/jsonschema-4.17.3-py311h06a4308_0/lib/python3.11/site-packages:$pkgs/pyrsistent-0.18.0-py311h5eee18b_0/lib/python3.11/site-packages:$pkgs/attrs-22.1.0-py311h06a4308_0/lib/python3.11/site-packages" \
  "$HOME/anaconda3/envs/mamba-ssm/bin/python" -m pytest -q
```

```powershell
# Windows PowerShell, from the repository root.
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
py -3.13 -m pytest -q
```

## Reproduce the CUDA queue

```bash
source "$HOME/.venvs/sigma-r1-cu121/bin/activate"
export PYTHONPATH="$PWD/src:$HOME/loopcrowd-lc002/deps"
export TORCH_HOME="$HOME/loopcrowd-lc002/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u scripts/run_lc003.py \
  --manifest "$HOME/loopcrowd-lc002/shanghaitech_a.jsonl" \
  --historical-train "$HOME/loopcrowd-lc002r/train/42" \
  --output "$HOME/loopcrowd-lc003-reproduction"
```

The queued Python 3.11.11 process measured one RTX 3090 with CUDA 12.1. Per-run CLI arguments, configuration hashes, timing, logs and checkpoint digests are recorded in [training history](training_history.json). The raw logs and checkpoint files remain local. The hosted matrix was skipped under the issue's offline resource limit.
