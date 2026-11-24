# LC-003 follow-up

Keep the fixed internal validation split, 0.30 primary cutoff and the 40-epoch budget for any follow-up. Preserve the schedule and seed eligibility rule when comparing these results.

If late validation variability remains, first inspect the local checkpoints' backbone BN running means and variances alongside epoch count totals. The first controlled extension would compare frozen backbone BN statistics with the existing train-mode BN using identical model initialization, crop order, LR and loss. Evaluate any crop-policy or slot-loss change separately, after that comparison.

Treat local nAP as a spacing-relative localization diagnostic. Official test evaluation remains deferred until the model and inference protocol are frozen. NWPU work requires separate owner authorization.
