# After LC-002 acceptance

1. Obtain and verify the original NWPU images. Reconcile coordinate rejections, audit train/val, and validate the authors' box-derived evaluator before reporting official NWPU F1.
2. Review the A/B crop sensitivity and the per-epoch overflow log. Choose capacity and any score calibration using only allowed train and validation evidence. Keep this pilot's scores and label policy traceable.
3. If a longer comparison is approved, freeze the optimizer, split, crop and stopping protocol across all five controls. Use multiple training seeds and compare four shared calls with four independent blocks at the recorded compute budget.
4. Run separately approved baseline and frozen test evaluations before drawing benchmark conclusions. A loop embedding or dynamic routing would require its own controlled experiment after this review.
