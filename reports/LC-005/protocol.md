# LC-005 frozen paired protocol

LC-005 starts from PR #12 head `06b2feb9faaa43f30e0fd322d3c9d3c7d54bbd59`. This file, the two configs and the code freeze before preflight. The experiment asks whether point supervision at the first shared call improves the second-call output on ShanghaiTech Part A internal validation.

## Matched arms

Both arms use one Shared-2 block with two calls. The first and second call features feed the same existing fuse, point logits and point offsets modules. The arms have 1,596,614 trainable parameters and use one backbone pass and two decodes per training batch. Both point losses participate in the backward graph. The optimized objective is final loss + λ × early loss. Final-only uses λ=0.0 and the auxiliary arm uses λ=0.25. Both follow the historical point target and focal classification plus offset regression losses. The default inference path decodes only the final output.

Pair order is seed 42 final-only then auxiliary, seed 123 auxiliary then final-only, and seed 2026 final-only then auxiliary. All six runs train from fresh, seed-matched initialization on the existing local ResNet18 IMAGENET1K_V1 cache. Backbone BN runs in train mode. Train270/internal-val30, random native-scale 512 px crops, overlap 128, batch two, stride four, two point slots, BF16, AdamW, weight decay 0.01 and the label policy are fixed. The 40-epoch horizon uses 135 updates per epoch and 5,400 per run. Warm-up is 270 updates followed by cosine, with peak LR 0.0002 and minimum 0.00002. Early stopping is disabled. The only config difference within a seed is λ.

## Preflight and evidence

Verify manifest, cached weights and the two config file digests. Compare the seed-specific initial state tensor hash including BN buffers, the ordered autograd node types and edges, parameter counts, warm-up crop identities, module calls and LR trajectories. The diagnostic batch uses the first two original training records at epoch one. Its disposable optimizer step is followed by three synchronized timed steps. Record warm-up device memory, allocator peaks and device/version metadata. Fresh training runs must match the preflight hashes and all 5,400 applied rates. The queued pairs must have identical hashes of the actual loader sequence of record indices and crop origins in every epoch. Each run logs attempts, successful updates and AMP skips. A memory or budget failure stops the queued experiment.

The LC-004B fit manifest is identified by its file hash for scope tracking. This experiment does no calibration. All checkpoints are selected by lowest validation MAE at the final output, with the first epoch winning ties. The early predictions are for diagnosis. The internal-validation candidate policy uses native-scale tiles, original strict score >0.05 and midpoint ownership for counting at strict score >0.30. Count-side local TP, FP and FN use one-to-one greedy matching at δ=0.50 on the owned count mask. Local nAP uses the original in-frame pool and local spacing-relative radii. Only original training and internal-validation image files are read.

## Endpoints and gate

Use each seed's arithmetic mean final MAE in epochs 36–40 as the primary endpoint. Report the paired auxiliary-minus-final-only differences, their mean and sample SD for the three seeds. Report analogous early MAE, RMSE and local nAP, final RMSE and local nAP, and best/final checkpoint MAE. The relative three-seed improvement divides the difference of the arithmetic mean arm MAEs by the final-only arithmetic mean.

The continuation gate requires all of the following.

1. At least 3% improvement in the three-seed mean final last-five MAE.
2. Lower final last-five MAE in at least two seeds.
3. Mean final last-five local nAP change of at least −0.005.
4. All six runs meet per-run stability. Final last-five MAE population CV is at most 10%, the absolute relative change in mean **final train loss** from epochs 31–35 to 36–40 is at most 3%, and the logged losses and gradient norms are finite.
5. A clear early-stage result means at least 3% improvement in the three-seed mean early last-five MAE and lower early last-five MAE in at least two seeds. Early local nAP and FP/FN provide separate diagnostics.

Analyze the final and early endpoints, their paired FP/FN changes, timing and allocator peak memory. LC-003 Shared-2 is historical context because its training graph used one decode. The new control matches the two-decode graph. The three seeds support an exploratory internal-validation comparison. The queue has six arms and ends at the prespecified budget.
