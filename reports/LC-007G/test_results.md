# CPU verification

The full suites ran the frozen implementation at `62b336bcf0bdeea9e5fa66ded2cd8c5e81fb4926` with CUDA hidden and existing dependencies. WSL used a native checkout from a local bundle.

| Python | Passed | Runtime | Local log SHA-256 |
| --- | ---: | ---: | --- |
| WSL 3.10.16 | 172 | 67.21 s | `85c87b9d36426d192a8744cda93826f6dbf5ca97863edeafa17cbbddd761702e` |
| WSL 3.11.11 | 172 | 67.90 s | `a4f6b96d95a935195d3345e56f87ae867c9d95ae4a8ac40e9b7f854db3899c4c` |
| Windows 3.13.2 | 172, nine deprecation warnings | 106.87 s | `a7e8e7e68a35a2e0ecae1beb87c540456793b05c186bc4cf72309eb12bace1b0` |

The new tests cover the four-call forward and gradient oracle, opposite-side isolation, padding-key independence on partial grids, exact FP32/BF16 stage-one outputs and stage-one-only gradients, midpoint isolation from calls 3/4, block identities, checkpoint round trip and one EMA shadow per unique parameter. Legacy scalar gain, gated, depth and final-only shifted paths pass the full regression suite. The uncapped guard tests verify available-device headroom without setting an allocator fraction, acceptance of observed reserved memory above 2 GiB, rejection of an approval that reintroduces a quota, and charged failed CPU preflight before CUDA work. [New tests](../../tests/test_cross_window.py)

The CPU audit verified 66 historical LC-007B/D/E tables, four fresh K=4 shifted starts against the full-state, decoder, block and RNG digests of the matched fixed references, and all 160 planned crop/LR pairs against historical actual pairs. The workspace and WSL-native freeze audits were identical at SHA-256 `52b8170975a9d1dbb04878f0682d4ac2f27d0bbcdf0f75756ddf82027f323efb`. [Audit data](initialization_audit.json)

The CPU result archiver recomputed all 2,560 fixed shift-step comparison rows from the indexed historical data and checked the subtraction order with a synthetic fixture. A stopped-queue fixture using a completed shifted checkpoint verified 1,280 stagewise rows and preserved the unevaluable status of an incomplete queue. [Archiver](../../scripts/archive_lc007g.py)
