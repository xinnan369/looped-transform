# LC-007C archive and gradient-probe protocol

Issue #21 authorizes archived CPU recomputation, generated CPU fixtures and preparation. The stacked parent is PR #20 head `0aeaea0562f075fd28ecaa48c6400193a7cf22f4`. The LC-007B execution freeze was `6471f222dbe1480ff7fca1ef0291215f2498d32f`. Current GPU status is `NOT_RUN_PENDING_OWNER_GPU_APPROVAL`.

## Frozen archive endpoint

Validate the SHA-256 of the original index and preflight, then validate all 24 indexed split files, byte and row counts. Reconcile E36–40 score/matching counts against 40 aligned validation IDs and fixed train-defined density groups. For each seed 42 and 123, policy, epoch and group, pair early and final at the same checkpoint in each architecture, then compute `(Shared final − early) − (Unshared final − early)`. Compare the finalists separately with independent Shared-1 and CNN under the same policy and seed. Report equal-weight per-seed epoch means and macro means. Keep E40 separate. Image-level records enter only as aligned descriptive comparisons across repeated views of the same development images.

Use the stored strict >.30 score and ownership for count, local greedy matching at δ .50, >.05 raw AP floor and exact archived pooled local nAP. Compute MAE and RMSE directly from the archived image counts for reconciliation. Use `signed = FP − FN` and `absolute = FP + FN − 2 min(FP,FN)` per image. Average the stagewise changes in absolute error across images. Explain changes in the number of missed and extra predictions and the offsetting contribution from FP/FN on the same image. Archive records contain no candidate or matched GT identities. Group nAP comes directly from the saved pooled scores.

## Frozen inputs for a separate GPU decision

Select exactly 12 batch-two, 512 px train crops per seed, grouped as four low, four mid and four high batches. Generate one native random crop origin per original training image from the LC-007B policy at fixed epoch 40. Compute heads per valid crop megapixel from annotation coordinates. Use linear train-crop tertiles per seed with lower-inclusive mid/high groups. Rank within each group by SHA-256 of the tag, seed, image ID and origin. Select eight distinct images and batch consecutive pairs. One seed's ordered crops are identical for both arms and both parameter policies. The [probe manifest](probe_manifest.json) fixes the exact seeds, group populations, boundaries, coordinates, ordered batch hashes and four E40 checkpoint references. Preparation uses only annotation metadata and bytes for checksum verification.

If separately approved, use Shared-2 aux and same-supervision Unshared-2 aux at E40, seeds 42 and 123. Verify the checkpoint, initial weight, manifest and effective config SHA-256 against the proposed manifest before using the already verified bytes. Load checkpoint state without updates. Use live and FP32 EMA parameter mappings from the same checkpoint and the same exact live BN starting buffers. Reset the cloned buffers and the RNG to the same starting snapshot before every forward. Preserve train-mode BN behavior during the BF16 loss forward and restore flags, RNG, original parameters, buffers and existing gradients after success or exception. Validate optimizer/scheduler/EMA step and checkpoint bytes as read-only. Do not schedule validation inference or any optimizer, EMA or scheduler step. E40 gradient evidence describes the saved endpoint.

### Gradient comparisons

- Calculate early and final supervision gradients on the same physical block or decoder basis. Measure their cosine and both L2 norms in FP32, the raw norm ratio and the norm ratio with λ=.25 on the early vector. Report `(.25 g_early) · unit(g_final)` and `g_final · unit(.25 g_early)`. Flag cosine below zero as a directional conflict. Use `no_gradient_path`, `zero_or_near_zero_norm` at ≤1e−12 or `nonfinite` as applicable, with a missing cosine. The second independent block has no early-loss path.
- Use a native graph and an expanded stateless graph initialized at equal parameter values. The two expanded block calls consume the same continuously differentiable trajectory and decode through equal-valued head copies. Compute gradients for final loss and total `Lfinal + .25 Learly` separately. Require early/final output and loss equivalence and verify the sum of both expanded call gradients against the native shared gradient for the block and decoder. For independent blocks check each on its own parameter object. Comparing their call vectors descriptively aligns names and shapes across different parameter spaces. The first decoder call has no path from final loss, while the first block does.
- Freeze numerical tolerances before project measurements. FP32 outputs and loss use absolute/relative `1e−6/2e−5`, and gradients `2e−6/2e−4`. BF16 outputs use `.002/.02`, loss `.0005/.01` and gradients `.001/.04`, with computed tensor differences and norms read in FP32. Record reachability and maxima per group/loss. Stop interpreting an expanded gradient whenever equivalence fails. Report distributions and directions separately by seed, crop-density group, policy and comparison type. Keep FP32 EMA weights paired with the saved live buffers in all cases.

### Execution guard and stop rules

The proposed limits are one serial RTX 3090, 96 batch-policy cases, ≤1,200 synchronized phase seconds, ≤2 GiB Torch process peak reserved memory, zero project updates and zero downloads. Proposed limits do not confer approval. Implement authorized runtime entry separately from the current CPU-only diagnostics. Synchronize and account for preflight, transfers, measurements and cleanup, including exception-path transfers and final synchronization. Include necessary cleanup inside the charged phase and reserve for it before entering. Propagate final synchronization failures, stop on resource/provenance/nonfinite failures, and record realized cases, time and peak allocation. The [CPU-tested phase accounting](../../src/loopcrowd/probe_budget.py) is cooperative and allocator-based. GPU timing and resource feasibility remain unverified. [CPU timing projection and numerical fixtures](gradient_equivalence.json)

## Reproduction on local CPU

```bash
source artifacts/lc002-env.sh
export CUDA_VISIBLE_DEVICES=-1
python -m scripts.archive_lc007c
python -m scripts.prepare_lc007c
python -m scripts.verify_lc007c_cpu
```

The environment script activates the existing `~/.venvs/sigma-r1-cu121` and local dependency paths. No package installation is required. All commits use `[skip ci]`.
