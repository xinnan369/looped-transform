# LC-007A frozen diagnosis protocol

The parent is PR #16 head `7848efc8dd572edb5de43b2f3aaa0e5eff46f209`. This is an inference-only replay and historical evidence audit. Keep the original experiment decisions and checkpoints intact. A/B internal validation has informed previous experiments and remains development evidence.

## Frozen cases

Each row uses seeds 42, 123 and 2026, and the archived epoch-40 `last.pt`. A contains 15 checkpoints and B contains 18. The frozen case index resolves every checkpoint digest before reading any validation image. Missing E40 files are marked `not_run`.

| Dataset | Training archive | Arms | Output in this audit |
| --- | --- | --- | --- |
| A | LC-003 warmup cosine | CNN, Shared-1, Unshared-2 λ0 | Final |
| A | LC-005 | Shared-2 λ0 and λ.25 | Early and final |
| B | LC-006 | CNN, Shared-1 | Final |
| B | LC-006 | Shared-2 λ0 and λ.25, Unshared-2 λ0 and λ.25 | Early and final |

Use one train/val-scoped manifest and at most one full internal validation pass per checkpoint. All dual reads in the table use a single staged forward per tile. The historical A replay/calibration artifacts lack matched distances and proximity coverage on the required fixed views. Use cached local images and checkpoint states, with pretrained initialization disabled. Test the staged final output on synthetic data. The A Unshared-2 λ.25 cell is absent. The A Unshared-2 λ0 comparison crosses LC-003 and LC-005, and is labeled as such.

## Frozen statistics

- Count uses ownership and a strict score greater than .30. Rank all in-frame candidates with score greater than .05 for pooled local nAP at δ .05 through .50. Use the historical greedy per-image matcher, stable score ties and the mean of up to three nearest GT neighbors for local normalization. TP/FP/FN and accepted pair distances use the count view at strict δ .50.
- Density is GT heads per native megapixel. Use train-only linear quantiles at 1/3 and 2/3 for each dataset. Tertile groups are `[-∞, q1)`, `[q1, q2)`, `[q2, ∞)`. Report dimensions, tile geometry, nearest other head distances and raw/owned/counted score histograms with edges .05, .10, …, 1.00.
- Coverage is the fraction of GT with at least one candidate within strict δ .50 before one-to-one assignment, separately for raw, owned, and count views. It is an upper bound on matching recall for that view. Measure candidates, GT, FP and FN within 4 px of an internal ownership line. The stride-4 K=2 fixed-origin frame grid overflow is only a capacity proxy. Training crop origin and tile padding shift the grid.
- Predefined final comparisons are Shared-2 λ.25 versus Shared-2 λ0, CNN, Shared-1 and, on B, Unshared-2 λ.25; Shared-2 λ0 versus Unshared-2 λ0 on both datasets; Unshared-2 λ.25 versus λ0 and CNN on B. Also compare each staged final with its own early readout. Within-checkpoint and independent-training results are separate. Report target minus reference, groups and all images, exact pooled nAP for each group, and improvement/regression/tie based on per-image absolute count error. Mark the five largest per-run absolute errors with deterministic ID tie breaking.
- Bootstrap 2,000 rows of image indices with NumPy PCG64 seed 7007, one fixed matrix per dataset shared across all seeds and contrasts. Average each image's paired error delta across seeds before descriptive percentile calculation. Leave-one-image-out uses the same omitted image for all paired runs. Report training-seed sample SD separately from image sensitivity. Neither is a significance test or a model selection rule.

## Execution

```bash
source artifacts/lc002-env.sh
PYTHONPATH="$PWD/scripts:$PYTHONPATH" python scripts/run_lc007a.py --output "$HOME/loopcrowd-lc007a"
```

After the run, use the tracked archival tool against that local directory. New optimizer updates, threshold fits and official test passes are zero. External experiment downloads, dependency installation, and NWPU/WiderPerson activity are out of scope.
