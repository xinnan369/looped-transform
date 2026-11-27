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

## Dedicated E40 driver verification

Driver freeze `f52546b39a15f6419568691c2777d2645392b1be`, execution config SHA-256 `2309902dbe1d059981ec190635dd55bd461a2dff27ebd3cb012539fe495cb064`.

| Environment | Full pytest result | Local log SHA-256 |
| --- | --- | --- |
| WSL Python 3.10.16 | 184 passed in 68.01 s | `d5f888d9d1daacbf787b12d64c90b252e2f821b42456654cf6140295c16e593e` |
| WSL Python 3.11.11 | 184 passed in 68.96 s | `3bc7497782e965a9bd55f7ac6d7c6bbab20709da7d5bd47e75e196c021a4b29b` |
| Windows Python 3.13.2 | 184 passed in 154.62 s, 9 dependency/deprecation warnings | `ab2723dfc28617b04d6a0e6605f58a3ebd5bc637bba90cdffe9fa787afee62fa` |

The four additional cases cover CPU checkpoint reconstruction and integrity rejection, count/rank/AP parity and drift rejection, one four-stage read per synthetic tile with all GT represented, explicit empty bins and constant placebo maps, duplicate-archive rejection and changed-config authorization rejection.

The CPU-only preflight rebuilt the four local E40 EMA models with live BN, confirmed 160 archived image rows and 16 grouped rows per model, and verified RNG parity. It did not decode real images. The clean WSL checkout reproduced the preflight SHA-256 `5c4819d7a5eed3fe8cef16d38487a728f6b27911448232306903a6750fecc299`.

| Checkpoint | CPU reconstructed state SHA-256 |
| --- | --- |
| 42 / S4-shift | `659c3a68f0337bac52639291cc8a9ef34340c5ab46f63b93a1059da5e9728cd8` |
| 42 / U4-shift | `0f6729e58ec6a770e9f9e01f76575dfa9a2c77fae5ed722c02cd929b46d3e672` |
| 123 / U4-shift | `91b3402c83863997fe4061254a8ba029065a421d4ddf9bb68babc704978d129c` |
| 123 / S4-shift | `af90780d43af6407472de1b3163308ebf6cca644216c638a4aaec04fa9e532a7` |
