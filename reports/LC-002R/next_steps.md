# After LC-002R review

1. Review the ownership ambiguity counts and AP fixture traces. If an author-provided P2PNet executable becomes available, compare ranking, assignments and AP on the saved fixture, then on identical validation predictions.
2. Obtain the remaining NWPU-Crowd image shards. Reconcile train/validation image bounds and annotations, then verify the official box-based F1 evaluator on the original data.
3. Use the late training loss, validation variability and sampled gradients to decide the next controlled training schedule. Change one training factor at a time and keep the fixed-window and CNN controls on the same budget.
4. Freeze the model and score cutoff using internal validation before any official-test run. Review LC-002R before considering LC-003.
