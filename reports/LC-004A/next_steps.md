# LC-004A follow-up

At seed 42 stage gain reached late MAE 223.75 and local nAP 0.17244. Tied gain reached 219.27 and 0.17635. The continuation checks in [the result summary](status.md) govern the stop decision.

A falsifiable follow-up is that shared recurrence changes score calibration more than the ordering of localizable candidates. For each frozen checkpoint, derive a score-only calibration mapping from a fit set carved out of the original training labels. Evaluate once on the frozen internal validation split. Test whether the mapping reduces the two-call MAE gap against Shared-1 while leaving local nAP materially unchanged. A persistent count gap after calibration or a simultaneous localization decline would weaken the hypothesis.

The analysis uses internal validation, 0.30 count cutoff and the 0.05 local nAP floor. NWPU and WiderPerson remain paused under the existing resource restriction.
