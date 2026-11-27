# CPU verification

Implementation freeze `9d0272df4b801c9caef4410b6566fb302e60ae53`. All runs disabled CUDA visibility and used existing environments and caches.

| Environment | Full pytest result | Local log SHA-256 |
| --- | --- | --- |
| WSL Python 3.10.16, `mamba-ssm` | 180 passed in 65.20 s | `1339abba5d9344da24c1a8589a93ba0d32f7631dfe708815829019d90d586be3` |
| WSL Python 3.11.11, `sigma-r1-cu121` | 180 passed in 66.00 s | `97dc7e4347c857bb3bcab7a670c58630ef138333f2a4784f66ee503865b334b1` |
| Windows Python 3.13.2 | 180 passed in 106.05 s, 9 dependency/deprecation warnings | `169ca095c0a27607b01b08e254b36ff8b813be94ba0b1a224224659529044cc9` |

The eight added cases cover odd-grid and padding alignment, max-slot aggregation, finite bounded centered gains, exact spatial placebo multiset, absence of cross-sample influence, detached confidence, residual scaling and exact unity behavior, FP32/BF16, matched initial tensor/RNG identities, t1/t2 parity, early gradient boundaries, ordinary state gradients, legacy/no-op loss and gradient equality, training context preservation, final/two/four read outputs, candidate and GT identity parity, and count/ranked AP parity. Inherited suites ran all earlier paths, including guarded time and memory resource accounting.

The CPU audit verified 78 archive-indexed files, exact initialization for six reference-paired configurations, and 7,200 planned crop/LR updates per case against the recorded reference trajectory. It verified four local checkpoint files and their provenance without running them. Rerunning the audit at the implementation freeze produced the same SHA-256 `33f599a55c82a4a745e4948d8fc34936f324e02a5e8e3b4bd5c0c05039525970`. [Initialization and cost audit](initialization_audit.json)

Run from the repository root with the existing local environment and `PYTHONPATH` pointing to `src`.

```bash
export CUDA_VISIBLE_DEVICES=-1
python -m pytest -q --tb=short
python -m scripts.diagnose_lc007h
python -m scripts.prepare_lc007h --manifest "$HOME/loopcrowd-lc002/shanghaitech_b.jsonl" --output artifacts/lc007h-initialization.json
```
