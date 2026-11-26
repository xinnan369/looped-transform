# Real-data stagewise results

Each shift-step difference is the shifted model's stage change minus the corresponding fixed model's stage change within the same epoch, seed, policy and density group. The table summarizes EMA E36–40 with equal seed weights. Negative MAE values favor the shifted stage change and positive local nAP values favor it. [All 2,560 matched shift-step comparisons](paired_shift_steps.csv)

| Shifted arm | Step | Density | ΔΔ MAE | ΔΔ local nAP |
| --- | --- | --- | ---: | ---: |
| S4-shift | 1→2 | All | +11.7400 | +0.003110 |
| S4-shift | 3→4 | All | −0.3800 | −0.001677 |
| U4-shift | 1→2 | All | −4.2600 | −0.008701 |
| U4-shift | 3→4 | All | +8.3150 | −0.002523 |
| S4-shift | 1→2 | High | +21.4545 | +0.002644 |
| S4-shift | 3→4 | High | −5.2091 | −0.000521 |
| U4-shift | 1→2 | High | −16.3455 | −0.001620 |
| U4-shift | 3→4 | High | +2.3364 | −0.002303 |

S4-shift improves final accuracy against its fixed control, with a larger nAP gain at step 1→2 and a smaller nAP gain at 3→4. At 3→4 the count MAE still rises by 0.1850 for S4-shift and 0.5650 for fixed S4. These changes compare stage trajectories from separately trained models, so they do not isolate the causal contribution of an individual shift call.

The archive includes MAE/RMSE, signed count error, TP/FP/FN, per-image counts, and state and update RMS. Count cancellation is twice the lesser of FP and FN per image. State and update RMS aggregate real-token means of squares over tiles and images. [All stage corrections](stagewise_correction.csv) · [Indexed inputs](epoch_index.csv)

Seam-localized recall, candidate identities and GT matched transitions are `not_available`. The decoder retains tile IDs, coordinates and scores while discarding the cell/slot IDs needed for identity comparisons. Validation used the unchanged matcher and one sweep per policy. The synthetic probe verifies attention-window connectivity. Boundary-specific recall remains unmeasured. [Token mechanism](cross_window_mechanism.md) · [Protocol](protocol.md)
