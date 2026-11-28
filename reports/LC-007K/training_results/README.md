# LC-007K single-seed crop and sharing result

`COMPLETE / READY_FOR_REVIEW`. The one new U4-shift-768 run completed 40 epochs and 7,200 optimizer and EMA updates. CPU archival verified the checkpoint, initialization, crop and LR hashes, and 40 epochs of actual matched exposure against S768. The [epoch index](epoch_index.csv) covers 14,120 training, grouped metric and per-image records. All cells use the same 40 B validation images and fixed 512 evaluation tiles.

## E36–40 final EMA

| Seed-42 model and training crop | MAE | Local nAP | RMSE | TP | FP | FN | Signed error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S512 | 23.875 | .305537 | 31.815 | 2,552.6 | 1,351.0 | 1,742.4 | −391.4 |
| U512 | 20.255 | .307775 | 27.284 | 2,631.6 | 1,404.8 | 1,663.4 | −258.6 |
| S768 | 19.700 | .346196 | 25.968 | 3,066.8 | 1,760.6 | 1,228.2 | +532.4 |
| U768 | 19.160 | .351294 | 25.974 | 3,070.2 | 1,730.0 | 1,224.8 | +505.2 |

The U768 crop comparison passed the primary MAE and nAP thresholds, the RMSE/FN guards and the 10% MAE CV guard (observed 4.56%). FP rose 23.15% and final loss changed −4.97%, beyond the 5% FP increase cap and the 3% absolute loss-change limit. The preregistered joint result is **fail**. The decline in loss is descriptive and does not by itself demonstrate instability.

The S−U MAE gap decreased from +3.620 at 512 to +0.540 at 768. The S−U nAP gap changed from −.002238 to −.005098. Crop effects S768−S512 and U768−U512 were −4.175 and −1.095 MAE, and +.040659 and +.043519 nAP. The descriptive difference-in-differences is −3.080 MAE and −.002860 nAP. S768 failed both strict metric directions versus U768 and failed its historical stability guard. Its original crop joint result remains fail. These comparisons use one seed and non-contemporaneous controls. Shared-2 aux+EMA remains the formal baseline. [Exact grid and checks](research_decision.json)

U768's final E40 EMA was 19.975 MAE and .353257 nAP. Its final live E36–40 was 24.305 MAE and .352135 nAP, and live E40 was 32.150 and .356093. These are secondary endpoints. The [endpoints](endpoints.csv), [grouped stage and density metrics](metrics.csv), [per-image counts and cancellation](images.csv), and [stage changes](stagewise_correction.csv) preserve the complete comparisons available from the frozen evaluation.

The [resource audit](resource_audit.md), [runtime preflight](preflight_cpu.json), and [phase and artifact evidence](run_evidence.json) document execution and provenance. Checkpoints and raw logs remain local and are indexed by path, size and SHA-256.
