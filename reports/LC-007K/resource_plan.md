# Single-attempt resource freeze

Proposed one RTX 3090 case, 40 epochs, 7,200 updates, 1,800 charged seconds, zero disposable GPU updates and zero external downloads. A dedicated launcher enforces TERM at 1,795 s and a five-second hard-kill grace period. CPU preflight, device initialization, transfers, validation, checkpointing, logging, gaps, failure and cleanup count toward the total.

| Phase | U512 historical seconds | S768 observed seconds | U768 planned seconds |
| --- | ---: | ---: | ---: |
| Training | 299.369 | 578.415 | 673.580 |
| Live validation | 308.068 | 296.558 | 308.068 |
| EMA validation | 310.369 | 284.369 | 310.369 |
| Initialization | .208 | .902 | .902 |
| Checkpoint | 2.990 | 1.751 | 2.990 |
| Base projection | | | 1,295.909 |
| 25% allowance plus 120 s | | | 443.977 |
| Total | | | 1,739.887 |

The plan takes the phasewise maximum of observed S768 and U512 projection. U512 training alone is multiplied by the larger of the dense and area ratios. The 1,174.758 s S768 charged total is historical. U768 runtime and training peaks remain unmeasured. The candidate fits the proposed 1,800 s ceiling with about 60 s remaining after allowances.

At an authorized launch, confirm 3 GiB local disk and actual device headroom against the 4,426,039,296-byte historical peak-based estimate. The runner observes before/after device memory and actual process peaks, leaves other jobs running, and applies no VRAM quota or allocator fraction. It stops and retains available evidence on OOM, nonfinite, time, data or trajectory errors, with one attempt and fixed batch and precision.

The current attempt is bound to the owner's direct launch instruction for Issue #36. The local authorization records the absence of independent review at launch. The earlier LC-007I waiver is still rejected.
