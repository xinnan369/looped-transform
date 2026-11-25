# LC-006 verification and reproduction

The protocol, configs and training code froze at `3929926602df3ef1af1ac22508bba27e728a8117`. Tests and training used existing local interpreters and installed packages.

| Interpreter | Exact test selection | Result |
|---|---|---|
| WSL Ubuntu 22.04, Python 3.11.11, torch 2.2.2+cu121, torchvision 0.17.2+cu121 | `pytest -q` | 89 passed in 158.07 s |
| WSL Ubuntu 22.04, Python 3.10.16, torch 2.3.1+cu118, torchvision 0.18.1+cu118 | `python -m pytest -q` with existing pure-Python schema packages | 89 passed in 161.34 s |
| Windows, Python 3.13.2, torch 2.11.0, torchvision 0.26.0 | `py -3.13 -m pytest -q` | 89 passed in 66.74 s with four compatibility deprecation warnings |

The two focused files passed nine tests in 143.15 s. Fixtures check block and storage independence, the shared decoder, parameter count, direct gradients at λ=0 and λ=0.25, staged loss reporting, default final inference, checkpoint loading/resume and skipped official-test file checks. The λ=0 final loss matches exactly on CPU and the gradient check uses relative tolerance 1e−6 and absolute tolerance 1e−7. The independent model's early loss contributes to its first block and existing head. Its second block receives the same gradient as the zero-weight arm at that matched initialization.

## Local commands

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

## B audit and CUDA queue

```bash
source "$HOME/.venvs/sigma-r1-cu121/bin/activate"
export PYTHONPATH="$PWD/src:$HOME/loopcrowd-lc002/deps"
export TORCH_HOME="$HOME/loopcrowd-lc002/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
work="$HOME/loopcrowd-lc002"
python -u -m scripts.audit_lc006_data \
  --manifest "$work/shanghaitech_b.jsonl" \
  --root "$work/data/part_B_final" \
  --rejections "$work/shanghaitech_b_rejections.json" \
  --output "$HOME/loopcrowd-lc006-data-audit.json"
python -u -m scripts.run_lc006 \
  --manifest "$work/shanghaitech_b.jsonl" \
  --root "$work/data/part_B_final" \
  --rejections "$work/shanghaitech_b_rejections.json" \
  --output "$HOME/loopcrowd-lc006-reproduction"
```

The queue reproduces the audit from local files before training. It validates the original split and post-cleanup labels, processes the documented seed order on one RTX 3090 and writes each run's raw artifacts outside Git. The [preflight](preflight.json) and the [per-run index](training_history.csv) link commands, configuration, checkpoint and history hashes. Hosted CI is outside the authorized resource scope.
