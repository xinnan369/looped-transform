# LC-007H status

`E40_DRIVER_FROZEN / GPU_REVIEW_PENDING`

CPU preparation completed against reviewed PR #30 head `4a0f7b023e314af8ac6fe3c8ea0390d3428dee63`. The CPU work includes the verified archived count diagnostic, candidate and GT identity export, guided and spatial-placebo variants, and the staged resource proposal. The dedicated [E40 execution driver](../../scripts/run_lc007h_e40.py) is frozen at `f52546b39a15f6419568691c2777d2645392b1be` for the diagnostic review requested on PR #32. Its offline checks verified the four exact checkpoint files and reconstructed their EMA parameters and live BN buffers on CPU. No checkpoint inference, GPU stage, training update, new experiment input download or hosted CI was initiated.

The archive shows S4-shift MAE 22.6650 versus U4-shift 20.7900, S2 21.9375 and U2 20.8575. S4-shift has higher pooled nAP than S2 and U2. Count errors and within-image cancellation explain why the count and localization results rank differently. Candidate- and GT-level uncertainty on real validation data is unmeasured.

Review the [archive](archived_error_diagnostic.md), [CPU tests](test_results.md), [protocol](protocol.md) and [read-only diagnostic proposal](resource_plan.md).
