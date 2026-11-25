# LC-007C CPU verification

Passed all three full local suites with CUDA devices hidden, on compatibility commit `2736668f0af3904cc2ac8a30be3381761365642a` after the protocol freeze. No packages were installed.

| Existing local environment | Result | Local log and SHA-256 |
| --- | --- | --- |
| WSL Ubuntu 22.04, Python 3.10.16, `mamba-ssm` | 111 passed in 29.46 s | `/home/xinnan/loopcrowd-lc007c-py310.log` · `010638c4a3dcb8220cfc0035614c948c690f6b964e7d8ad7d3f66847882e6d1a` |
| WSL Ubuntu 22.04, Python 3.11.11, `sigma-r1-cu121` | 111 passed in 28.24 s | `/home/xinnan/loopcrowd-lc007c-py311.log` · `77d7c34c354d5d2e062c14aeb55ee6abae7991120bc5058f12c52a9baa0aad96` |
| Windows, Python 3.13.2 | 111 passed in 72.59 s | `artifacts/lc007c-py313.log` · `8de9de47a8c010a6035aac1efeb90f60058b830d608233935c549860cdcca3b2` |

The initial Python 3.10 run stopped at collection because a starred index used 3.11 syntax. The follow-up uses a tuple expression compatible with 3.10. An earlier WSL 3.11 pass was interrupted while a `git status` subprocess waited on the Windows-mounted checkout. WSL re-verification used a local Git bundle checked out on the native WSL filesystem. The initial logs are retained locally as `loopcrowd-lc007c-py310-first.log` and `loopcrowd-lc007c-py311-mounted.log`.

The fixtures cover equal-valued two-call graphs, native gradient reconciliation, absent gradients, tiny and nonfinite norms, weighted projections, pseudo-forward exceptions, BN/mode/RNG restoration, unchanged live and EMA fixture tensors, train-only batch strata and final synchronization error propagation. The generated BF16 benchmark checked 64 px and 128 px batch-two cases for both graphs. The largest gradient element difference from summing the call gradients was 0.001953125 and passed the frozen absolute-plus-relative criterion. Synthetic gradients serve method validation. The project-gradient file expressly records zero executed cases.

Separately reproduced the two archive CSVs, integrity JSON, probe manifest and project status CSV byte-for-byte. Independently checked the cancellation and difference-in-differences identities in all 800 paired image rows, exact digests of the generated archive tables, and distinct ordered hashes of all 24 train-only batches.

From the repo root in the activated WSL environment, run `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q --tb=short`. The 3.10 suite reused the locally cached jsonschema, pyrsistent and attrs modules on `PYTHONPATH` as in the LC-007B verification. Windows uses `py -3.13 -m pytest -q --tb=short` with `CUDA_VISIBLE_DEVICES=-1` and two OMP/MKL threads. Hosted CI is `not_run`.
