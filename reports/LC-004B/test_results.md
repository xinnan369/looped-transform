# LC-004B verification and reproduction

Code and fit population froze at `a7b5d14be80aad4ac73723661328071cb7570ca1`. The tests and diagnostic used existing environments.

| Interpreter | Command | Result |
|---|---|---|
| WSL Ubuntu 22.04, Python 3.11.11, torch 2.2.2+cu121, torchvision 0.17.2+cu121 | `pytest -q` | 80 passed in 77.56 s |
| WSL Ubuntu 22.04, Python 3.10.16, torch 2.3.1+cu118, torchvision 0.18.1+cu118 | `python -m pytest -q` with cached pure-Python schema packages | 80 passed in 78.41 s |
| Windows, Python 3.13.2, torch 2.11.0, torchvision 0.26.0 | `py -3.13 -m pytest -q` | 80 passed in 41.45 s, with the existing AnyIO and GradScaler deprecation warnings |

The seven new calibration cases cover deterministic train-only stratification independent of record ordering, strict equality at cutoffs and the float32-to-float64 boundary, both cutoff tie-breaks, local match prefixes, provenance-sensitive candidate hashes, bitwise repeat of fixed-pool AP, monotonic mapping at both grid endpoints and the three-seed gap gate. The first targeted run passed all seven in 22.20 s. The real run separately verifies each fixed raw baseline against LC-004A. Each of the 18 cases receives one fit and one validation evaluation. Rank and candidate invariance checks reuse that evaluation's arrays.

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

## Frozen diagnostic

```bash
source "$HOME/.venvs/sigma-r1-cu121/bin/activate"
export PYTHONPATH="$PWD/src:$HOME/loopcrowd-lc002/deps"
export TORCH_HOME="$HOME/loopcrowd-lc002/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m scripts.run_lc004b \
  --manifest "$HOME/loopcrowd-lc002/shanghaitech_a.jsonl" \
  --lc003 "$HOME/loopcrowd-lc003" \
  --output "$HOME/loopcrowd-lc004b-reproduction"
```

The runner regenerates the fit selection in memory and checks it against the committed manifest. Checkpoint digests and validation outputs are reproducible through the command above with a new output directory. The exact observed command, CUDA environment, timing and raw file digests are in [the evidence index](calibration_evidence.json). Hosted CI is outside the authorized resource scope.
