# One native training crop comparison

Execution freeze `7bf410956113184493b0d101694873b7b6219ac5`, Git tree `b450f22f8a3e36e88ef1030adc74109bd35d4230`, execution config SHA-256 `f14c9befb3f7a434f82f723a54831f00eef3c21d8b4be0789dd9c7840aed80e8`. The dedicated authorization scope is `LC007I_single_seed42_native768`. [Frozen configuration](../../configs/lc007i_native_crop.json)

The single fresh run uses Shifted-Shared-4 at seed 42, 40 epochs of 180 successful updates each, and cached ShanghaiTech B train360/internal-val40. Training uses native 768×768 crops, no rescaling, batch 2, BF16, train BN, width 192, six heads, eight-token windows shifted 0/4/0/4, two point slots and 1,596,614 parameters. AdamW uses .0002 LR, .01 weight decay, 360-step warmup and cosine decay to .00002. FP32 EMA .99 averages unique parameters and uses live BN buffers. Use the final loss and .25 midpoint loss with exactly two training decodes.

Evaluation retains original 512×512 tiles with overlap 128. Each epoch performs one four-stage sweep with live weights and one with EMA. Count uses owned candidates with score >.30 and δ=.50. Pooled local nAP uses the existing >.05 floor and δ=.05:.05:.50. Density strata use the frozen train-only cutpoints. This change preserves validation geometry. The nominal training crop area increases 2.25×. Pixel exposure, sampled GT and computational cost are allowed to differ.

## Prespecified acceptance

Compare E36–40 final EMA against the historical same-seed S4-shift 512 run. Historical MAE is 23.875, nAP .30553705790378227 and RMSE 31.81492807782338. Require MAE ≤23.15875, strictly higher nAP, sample MAE CV ≤10%, absolute relative final loss change from E31–35 to E36–40 ≤3%, RMSE ≤1.03×, and FP/FN each ≤1.05×. Report E40 and live as secondary endpoints, plus density and stage metrics, signed and absolute errors, TP/FP/FN and cancellation. This non-contemporaneous, single-seed comparison is exploratory and cannot establish a sharing advantage. Keep the thresholds, epochs and readouts fixed even on a negative result.

The runner checks actual crop-size-aware and image-order hashes, LR and exposure per epoch. The archiver verifies checkpoint, EMA, denominators and per-image metrics and records available artifacts on an interrupted attempt. [Runner](../../scripts/run_lc007i.py) · [Launcher](../../scripts/run_lc007i.sh) · [Archiver](../../scripts/archive_lc007i.py)
