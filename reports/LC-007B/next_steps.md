# LC-007B handoff

The Stage 1 stacked PR is for owner acceptance. Review the full live/EMA and architecture results against the frozen decision file. The high-density analysis reports development-validation gaps within identical readout policies and separates changes in stability from architecture accuracy.

Any expansion requires separate approval and a new frozen scope. The planning envelope is 60 runs, five seeds and 378,000 updates across A and B, with seeds 42, 123, 2026, 31415 and 27182. Exact reuse of all eight completed Stage 1 runs would leave 52 runs, 320,400 updates and approximately 7.27 historical-model RTX 3090 hours. These estimates apply only if the saved evidence meets the future freeze and the owner approves proceeding. The planned B training reuse removes an estimated 1.05 hours from the prior 8.32-hour model. It does not reduce future validation of the reuse conditions.

For a future joint accuracy claim, use the same readout policy for all arms, the 3% count target against CNN and independent Shared-1 and a 3% challenge against matched Unshared-2 on each dataset. Check at least three of five favorable paired seeds, positive mean nAP for a joint gain, RMSE ≤1.03× its comparator, and each of FP and FN ≤1.05×. Keep the prior EMA stability requirements for every main comparison. The proposed nAP tolerance is −.005 for exploratory non-regression. The final test protocol requires separate approval after model, data policy and thresholds freeze. Local nAP has no official parity, and NWPU/WiderPerson remain outside this scope.

[Authorization and frozen scope](https://github.com/xinnan369/looped-transform/issues/19) · [Stage 1 decision](research_decision.json)
