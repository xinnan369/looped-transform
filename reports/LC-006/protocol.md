# LC-006 frozen ShanghaiTech B replication protocol

LC-006 starts from PR #14 head `0588f3d93943a392b6c07fe124a4d745fa81a43c`. It transfers the LC-005 training recipe to ShanghaiTech Part B using fresh seeded initialization and the existing ResNet18 cache. It adds a matched independently parameterized two-block control. No A model checkpoint initializes a B run. The primary endpoint is mean internal-validation MAE in epochs 36–40 at strict score >0.30.

## Dataset

Use the original LC-001/002 Part B manifest SHA-256 `de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482`. All 400 original training-image and MAT-label pairs are checked. Rebuild the frozen membership from the SHA-256 ranking of `42:train/IMG_n`, replay coordinate base zero and the recorded out-of-bounds rejection policy, and match the stored train360/internal-val40 IDs, points and counts. Abort on a mismatch or missing file. The 316 official test images are excluded from all file checks, evaluation and selection. Train defines the density tertiles. Check stride-four, two-slot capacity on both permitted splits and a reproducible 512 px crop probe.

## Six arms and ordering

CNN and Shared-1 each have one final prediction and no early loss. Shared-2 and Unshared-2 each have a matched λ=0.0 arm and λ=0.25 arm, both with two decodes and the same final-plus-weighted-early loss graph. The two stages use the same existing fuse, classification and offset modules. Shared-2 calls one block object twice. Unshared-2 calls two independently initialized blocks of the same width and fixed window geometry. A/B arms of each two-call architecture differ only in λ for a given seed. All six preserve identical common and first-block initialization where applicable and the same actual crop sequence for that seed.

| Seed | Ordered arms |
|---:|---|
| 42 | CNN, Shared-1, Shared-2 final-only, Shared-2 aux, Unshared-2 final-only, Unshared-2 aux |
| 123 | Shared-1, CNN, Unshared-2 aux, Unshared-2 final-only, Shared-2 aux, Shared-2 final-only |
| 2026 | CNN, Shared-1, Shared-2 final-only, Shared-2 aux, Unshared-2 final-only, Unshared-2 aux |

The six config files mirror LC-005 training settings, with explicit training audits and the documented variant and λ. Native 512 px random crops, overlap 128, batch two, two stride-four slots, BF16, train-mode backbone BN, AdamW peak LR 0.0002, weight decay 0.01 and fixed 40 epochs are shared. Disable density loss, residual gates, shifting, loop embedding, activation checkpointing and early stopping in this matrix. Each run has 180 updates per epoch, 7,200 updates and 360 warm-up updates before cosine to 0.00002. The queue has 18 runs and a 129,600-update budget. A resource or update-budget violation stops the queue.

## Measurement

Before training, verify disk artifacts, seed-specific model hashes, common and first-block hashes across applicable architectures, distinct independently drawn second blocks, A/B autograd graph hashes, decoder calls, crop probe and exact LR horizon. Pairwise actual crop hashes and rates are checked as each run finishes. Profiles use one RTX 3090, the same 512 px synthetic BF16 batch two warm step, nine synchronized repetitions and the same internal-validation frame. Inference timing and FLOPs use the default final decoder. Timing is descriptive under workstation load. Record parameter counts and allocated/reserved process memory.

Count evaluation uses midpoint ownership and strict score >0.30. Local TP/FP/FN apply spacing-relative, stable greedy one-to-one matching at δ=0.50 to the count mask. Local nAP keeps all in-frame tile candidates above the original 0.05 score floor. Select best checkpoints by earliest lowest final MAE. Archive E40 and best as supplemental endpoints. The primary late mean never depends on either checkpoint selection or a threshold sweep. Calibration and official test evaluation are outside this run.

## Research gates

- **Replication.** Shared aux relative to shared final-only must lower the mean of the three late final MAEs at least 3%, improve at least two seeds, have mean final local nAP change at least −0.005 and pass stability for all six shared A/B runs. Early late MAE must improve at least 3% by the same mean calculation and improve at least two seeds. Report the A result separately.
- **Supervision specificity.** Recompute the same paired effects for independent two-block models. In percentage points and heads, report the shared aux gain minus the independent aux gain. The comparison estimates a supervised-objective effect in each architecture.
- **Depth.** Compare the shared aux final endpoint with independently trained Shared-1 and CNN. To support further depth exploration, the final late MAE must improve on Shared-1 by at least 3% on the three-seed mean and improve at least two seeds. The mean local nAP margin is −0.005. The aux model also needs an observed within-checkpoint improvement from early to final in either counting or localization, operationalized as a favorable three-seed mean change and favorable direction in at least two seeds for MAE or local nAP. Report both and label any count–localization trade-off.
- **Sharing.** Compare shared and independent aux arms on the same two-call and supervision budget. Report paired mean differences and sample SD, parameter count, FLOPs, VRAM and measured latency. Interpret only the observed count, localization and resource trade-offs.

Stability per run requires late final MAE population CV ≤10%, absolute E31–35 to E36–40 mean final training loss change ≤3% and finite losses and gradient norms. All 18 arms are reported regardless of stability. A favorable effect is a lower MAE or higher local nAP, evaluated on three seeds only. Relative reductions divide the difference of the two three-seed means by the comparator mean. The study ends after this matrix.
