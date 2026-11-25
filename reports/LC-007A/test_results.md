# LC-007A local verification

The full suites ran from analysis and archive commit `cda7d822eea17f2b8f05864c5ff6f877e663940a`. The subsequent evidence commit changes report text only.

| Local environment | Result | Local log and SHA-256 |
| --- | --- | --- |
| WSL Ubuntu 22.04, Python 3.10.16, `mamba-ssm` | 94 passed, 193.99 s | `/home/xinnan/loopcrowd-lc007a-py310.log`, `71a06c860e4668b592b8dca4739882ab3fe103cee3456d3dc381eed3cae7696d` |
| WSL Ubuntu 22.04, Python 3.11.11, `sigma-r1-cu121` | 94 passed, 194.36 s | `/home/xinnan/loopcrowd-lc007a-py311.log`, `f027cc164c122258b1b47ecb49477a6e0bac411bca018ae315353067a2ee38e4` |
| Windows, Python 3.13.2 | 94 passed, 52.90 s | `artifacts/lc007a-py313.log`, `8e8af862e1566aa4aa8e82d02b12850cb10f16c6d113f3d8fc7042e25b1c66c3` |

The initial Python 3.10 run had 93 passes and one synthetic resume assertion during a branch commit. The fixture enforces an exact source commit across save and resume. Holding HEAD fixed resolved its result without changing training or test code. Initial log `/home/xinnan/loopcrowd-lc007a-py310-first.log`, SHA-256 `4fb167d7871214226e4dbf5c2e30ceb822834aedf5cb365e11a6e23d77139065`.

New tests check density edges, empty GT, accepted match distances, candidate pool invariance, complete and unique image pairing and deterministic reuse of bootstrap indices. Existing synthetic tests verify Shared-2 and Unshared-2 staged final candidates against the regular path, gradient routing and checkpoint resume. Existing training smoke fixtures use synthetic temporary data. The experiment itself used zero optimizer updates.

Independently verified 33 completed passes, all 51 archive reconciliations, all 102 split-file byte counts, hashes, row counts and image alignments, and per-image confusion-count and proximity inequalities. The archival command re-created 112 derived files byte-for-byte using the saved replay outputs.

## Reproduction in the existing environments

```bash
source artifacts/lc002-env.sh
python -m pytest -q
pkgs="$HOME/anaconda3/pkgs"
PYTHONPATH="$PWD/src:$pkgs/jsonschema-4.17.3-py311h06a4308_0/lib/python3.11/site-packages:$pkgs/pyrsistent-0.18.0-py311h5eee18b_0/lib/python3.11/site-packages:$pkgs/attrs-22.1.0-py311h06a4308_0/lib/python3.11/site-packages" \
  "$HOME/anaconda3/envs/mamba-ssm/bin/python" -m pytest -q
```

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
py -3.13 -m pytest -q
```

Hosted CI is `not_run` under the exhausted download budget. Every commit carries `[skip ci]`. Local pass results apply to the three recorded environments.
