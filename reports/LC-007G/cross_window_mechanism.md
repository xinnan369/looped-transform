# Synthetic cross-window mechanism

The probe feeds the same width-192 WindowBlock with the same seed-42 weights and token inputs, excluding the CNN and decoder. It injects a nonuniform channel displacement at (7,7) on a 17×19 token grid in one batch member. The comparison batch member stays unchanged. Reachability expands at each call to every real token that shares the current call's window with any previously reachable token. Key-masked padding contributes no graph nodes. This oracle bounds possible connectivity.

| Call | Fixed shift | Fixed oracle / observed | Shifted offset | Shifted oracle / observed | Gradient target token |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 64 / 64 | 0 | 64 / 64 | (7,7) |
| 2 | 0 | 64 / 64 | 4 | 144 / 144 | (8,8) |
| 3 | 0 | 64 / 64 | 0 | 256 / 256 | (15,15) |
| 4 | 0 | 64 / 64 | 4 | 323 / 323 | (16,16) |

The table agrees in FP32 and BF16 at the preregistered max-channel absolute thresholds of 1e−6 and 0.001 respectively. Maximum differences outside the oracle were zero, including opposite-side locations until a valid multi-call path reached them. The gradient of each indicated output's scalar probe reaches the perturbed token along the predicted path. In fixed mode, calls 2–4 show zero source gradient for the indicated cross-window targets. Actual norms and differences are in the audit. The observed coincidence with the oracle is specific to this generated input and weight state. [Probe code](../../scripts/probe_cross_window.py) · [Raw mechanism evidence](initialization_audit.json)

Additional tests cover 1×1, 3×11 and 17×19 grids and batch size two. Shifted partition/reverse is exact, no window containing real positions is fully masked, and changing padded keys leaves valid-query outputs unchanged. The existing CNN and decoder mix information spatially. This evidence characterizes direct attention-window connectivity and leaves real-data accuracy to the separate B screen. [Tests](../../tests/test_cross_window.py) · [Real-data plan](stagewise_analysis.md)
