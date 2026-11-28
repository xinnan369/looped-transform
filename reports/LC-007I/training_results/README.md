# LC-007I native crop result

`COMPLETE / READY_FOR_REVIEW`. The single seed-42 Shifted-Shared-4 run completed 40 epochs and 7,200 optimizer and EMA updates. CPU archival verified the frozen recipe, initialization, crop and image order, learning-rate trajectory, expected valid pixels and slot counts, checkpoint and validation coverage. The [epoch index](epoch_index.csv) contains checksums and row counts for 14,120 training, grouped and per-image records.

## Primary endpoint

The table uses the final EMA read at E36–40, with equal weight per epoch. The earlier 512 training run is the same seed and recipe, with non-contemporaneous execution. Both models use 512 validation tiles.

| Metric | 512 reference | 768 crop | Preregistered check |
| --- | ---: | ---: | --- |
| MAE | 23.8750 | 19.7000 | Pass, requires ≤23.15875 |
| Local nAP | .305537058 | .346196325 | Pass, strictly higher |
| RMSE | 31.8149 | 25.9676 | Pass, ≤1.03× reference |
| False positives | 1,351.0 | 1,760.6 | Fail, +30.32% vs ≤5% |
| False negatives | 1,742.4 | 1,228.2 | Pass, −29.51% |
| Sample MAE CV | | 5.70% | Pass, ≤10% |
| Relative final-loss change E31–35 to E36–40 | | −4.13% | Fail, absolute change ≤3% |

The joint decision is **fail**. MAE fell 17.49%, and nAP rose .040659. The larger crop increased true positives from 2,552.6 to 3,066.8, and signed count bias shifted from −391.4 to +532.4 predictions across the 40 images per sweep. MAE and nAP improved in each fixed density stratum. E40 EMA was 20.325 MAE and .345709550 nAP for 768, compared with 23.025 and .307818618 for 512. The single seed and non-contemporaneous design limit the result to an exploratory crop comparison. Shared-2 aux+EMA remains the formal baseline. [Decision checks](research_decision.json) · [Density and secondary endpoints](endpoints.csv)

## Artifacts

- [Training, EMA and checkpoint ledger](training.csv)
- [Live and EMA stage and density metrics](metrics.csv)
- [Per-image count, matching and cancellation](images.csv) · [Paired stage changes](stagewise_correction.csv)
- [Runtime input and exposure verification](preflight_cpu.json) · [Run evidence and local artifact digests](run_evidence.json)
- [Resource audit](resource_audit.md) · [Authorization summary with local record SHA-256](authorization.json)

Checkpoint binaries and raw runner logs remain local. The published authorization summarizes the owner's instruction and retains the digest of the verbatim local record.
