"""Train, validate and resume fixed-loop comparisons."""

import hashlib
import json
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .accuracy_diagnosis import density
from .calibration import count_match_metrics, json_digest, summarize_counts
from .data import TrainingCrops, load_rgb
from .diagnostics import LoopStatistics
from .ema import ParameterEMA, preserve_training_context
from .experiment import (
    DEFAULT_CONFIG, checkpoint, file_sha, load_checkpoint, make_model, provenance, seed_all,
    state_sha, weight_provenance, write_json,
)
from .inference import autocast, predict_image, predict_image_stages
from .grouped_validation import grouped_validation
from .manifest import read_manifest
from .metrics import SCORE_THRESHOLDS, count_metrics, point_nap, score_threshold_sweep
from .scheduling import UpdateSchedule
from .targets import point_loss


def gpu_scaler(device, precision):
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")


def optimizer_for(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


def supervised_point_loss(model, images, target, config, diagnostics=None):
    weight = config.get("early_loss_weight")
    if weight is None:
        return point_loss(model(images), target, config["density_weight"], diagnostics)
    predictions = (model.forward_depth_stages(images) if config.get("depth_screen") else model.forward_stages(images))
    early_parts = {} if diagnostics is not None else None
    early = point_loss(predictions["early"], target, config["density_weight"], early_parts)
    final = point_loss(predictions["final"], target, config["density_weight"], diagnostics)
    if diagnostics is not None:
        diagnostics.update({f"early_{key}": value for key, value in early_parts.items()})
        diagnostics.update(early_loss=early.detach(), final_loss=final.detach())
    return final + weight * early


def train_step(model, optimizer, scaler, images, target, config, device, diagnostics=None, *, require_finite=False):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device, config["precision"]):
        loss = supervised_point_loss(model, images, target, config, diagnostics)
    scaler.scale(loss).backward()
    if diagnostics is not None:
        scaler.unscale_(optimizer)
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        norms = torch._foreach_norm(grads)
        diagnostics["gradient_l2"] = torch.linalg.vector_norm(torch.stack(norms))
    if require_finite and not torch.isfinite(torch.stack([loss.detach(), *diagnostics.values()])).all():
        raise ValueError("nonfinite training loss or gradient before optimizer step")
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


def evaluate(model, records, manifest_dir, config, device, all_predictions=None, sweep=False, per_image=False,
             *, trace=None):
    model.eval()
    samples, count_samples, expected, observed = [], [], [], []
    summary = {}
    image_counts = []
    threshold = config["count_threshold"]
    if sweep and config["localization_min_score"] > min(SCORE_THRESHOLDS):
        raise ValueError("score floor is above the first validation sweep threshold")
    for record in records:
        with load_rgb(record, manifest_dir) as image:
            predictions = predict_image(model, image, config, device, trace=trace)
        if all_predictions is not None:
            all_predictions.append({
                "image_id": record.image_id, "dataset": record.dataset, **predictions.export(),
            })
        for key, value in predictions.diagnostics(threshold).items():
            summary[key] = summary.get(key, 0) + value
        if record.points is not None:
            owned = predictions.count_candidates()
            expected.append(len(record.points))
            observed.append(int((owned[:, 2] > threshold).sum()))
            if per_image:
                image_counts.append({
                    "image_id": record.image_id, "width": record.image_width, "height": record.image_height,
                    "expected_heads": expected[-1], "predicted_heads": observed[-1],
                })
            samples.append((record.points, predictions.candidates))
            if sweep:
                count_samples.append((record.points, owned))
    result = {
        "images": len(records), "labeled_images": len(expected),
        "count": count_metrics(observed, expected) if expected else None,
        "expected_heads": sum(expected) if expected else None,
        "predicted_heads_on_labeled": sum(observed) if expected else None,
        "nap_local_v1": point_nap(samples) if samples else None,
        "nwpu_official_f1": "not_ready",
        "candidate_diagnostics": summary,
        "nap_score_floor": config["localization_min_score"],
    }
    if sweep:
        result["score_threshold_sweep"] = score_threshold_sweep(count_samples)
    if per_image:
        result["per_image_counts"] = image_counts
    return result


def evaluate_outputs(model, records, manifest_dir, config, device, *, staged):
    model.eval()
    stages = ("early", "final") if staged else ("final",)
    samples = {stage: [] for stage in stages}
    counts = {stage: [] for stage in stages}
    cutoff = config["count_threshold"]
    for record in records:
        with load_rgb(record, manifest_dir) as image:
            outputs = (predict_image_stages(model, image, config, device) if staged
                       else {"final": predict_image(model, image, config, device)})
        for stage, predictions in outputs.items():
            owned = predictions.count_candidates()
            active = owned[owned[:, 2] > cutoff]
            matched, = count_match_metrics(record.points, active, (cutoff,))
            counts[stage].append({"gt": len(record.points), "matches": matched})
            samples[stage].append((record.points, predictions.candidates))
    result = {}
    for stage, rows in counts.items():
        metrics = summarize_counts(rows, "matches")
        result[stage] = {
            "images": len(records), "count": {key: metrics[key] for key in ("mae", "rmse")},
            "expected_heads": metrics["gt"], "predicted_heads_on_labeled": metrics["predicted"],
            "nap_local_v1": point_nap(samples[stage]), "nap_score_floor": config["localization_min_score"],
            "local_match_delta": 0.50,
            **{key: value for key, value in metrics.items() if key.startswith("local_")},
        }
    return result


def _epoch(model, optimizer, scaler, loader, config, device, schedule, *, ema=None, guard=None):
    model.train()
    losses = torch.zeros((), device=device)
    class_losses = torch.zeros_like(losses)
    offsets = torch.zeros_like(losses)
    staged = config.get("early_loss_weight") is not None
    stage_losses = {key: torch.zeros_like(losses) for key in (
        "early_loss", "early_classification_loss", "early_offset_loss", "final_loss",
    )} if staged else {}
    order = hashlib.sha256() if staged or config.get("audit_training") else None
    heads = overflow = images_seen = valid_slots = skipped = 0
    rates, gradient_norms = [], []
    scale = scaler.get_scale()
    for images, target in loader:
        if guard is not None:
            guard.check("train_batch")
        if order is not None:
            crop_identity = target.pop("crop_identity")
            order.update(crop_identity.numpy().astype("<i8", copy=False).tobytes())
        images_seen += len(images)
        heads += int(target["heads"].sum())
        overflow += int(target["overflow"].sum())
        valid_slots += int(target["valid"].sum()) * config["slots"]
        gpu_target = {key: value.to(device, non_blocking=True) for key, value in target.items()}
        diagnostics = {}
        rate = schedule.apply()
        loss = train_step(model, optimizer, scaler, images.to(device, non_blocking=True),
                          gpu_target, config, device, diagnostics, require_finite=guard is not None)
        next_scale = scaler.get_scale()
        updated = next_scale >= scale
        if updated:
            schedule.global_step += 1
            rates.append(rate)
            if guard is not None:
                guard.record_update()
        else:
            skipped += 1
        if ema is not None:
            ema.update(updated=updated)
        if guard is not None:
            if not updated:
                raise ValueError("Stage 1 requires every optimizer update")
            ema.require_finite(check_live=True)
            guard.check()
        scale = next_scale
        losses += loss
        class_losses += diagnostics["classification_loss"]
        offsets += diagnostics["offset_loss"]
        for key in stage_losses:
            stage_losses[key] += diagnostics[key]
        gradient_norms.append(diagnostics["gradient_l2"])
    mean_loss = float(losses / len(loader))
    gradients = torch.stack(gradient_norms).cpu().numpy()
    finite = gradients[np.isfinite(gradients)]
    if not np.isfinite(mean_loss) or len(finite) + skipped < len(gradients):
        raise RuntimeError("training loss or observed gradient norm became non-finite")
    positive_slots = heads - overflow
    staged_metrics = {key: float(value / len(loader)) for key, value in stage_losses.items()}
    if order is not None:
        staged_metrics["crop_order_sha256"] = order.hexdigest()
        if not staged:
            staged_metrics["final_loss"] = mean_loss
    return {
        "loss": mean_loss, "classification_loss": float(class_losses / len(loader)),
        "offset_loss": float(offsets / len(loader)), "crops": images_seen,
        "sampled_gt": heads, "overflow_gt": overflow, "overflow_rate": overflow / heads if heads else None,
        "positive_slots": positive_slots, "negative_slots": valid_slots - positive_slots,
        "gradient_l2_last_step": float(gradients[-1]) if np.isfinite(gradients[-1]) else None,
        "gradient_l2": {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std()) if len(finite) else None,
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "samples": len(gradients), "nonfinite": len(gradients) - len(finite),
        },
        "global_step": schedule.global_step, "optimizer_updates": len(rates), "amp_skipped_updates": skipped,
        "learning_rate": rates[-1] if rates else None, "learning_rates": rates,
        **staged_metrics,
    }


def _phase(guard, name):
    return guard.phase(name) if guard is not None else nullcontext()


def fit(config, manifest, output, device, resume=False, stop_after=None, *, guard=None, expected=None):
    config = DEFAULT_CONFIG | config
    records = read_manifest(manifest, splits=("train", "val"))
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    if not train or not val or len({record.dataset for record in train + val}) != 1:
        raise ValueError("fit requires one dataset with nonempty train and validation splits")
    end_epoch = config["epochs"] if stop_after is None else stop_after
    if not 1 <= end_epoch <= config["epochs"]:
        raise ValueError("stop-after must be inside the training horizon")
    if output.exists() and not resume:
        raise ValueError(f"new run requires a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=resume)
    staged = config["early_loss_weight"] is not None
    ema_enabled = config.get("ema_decay") is not None
    if guard is not None and not ema_enabled:
        raise ValueError("the Stage 1 resource guard requires parameter EMA")
    audited = staged or config["audit_training"] or ema_enabled
    with _phase(guard, "initialization"):
        seed_all(config["seed"])
        model = make_model(config, initialize=not resume)
        initial_sha = state_sha(model) if audited and not resume else None
        if expected is not None and initial_sha != expected["initial_sha256"]:
            raise ValueError("starting model differs from the frozen initialization")
        model.to(device)
        ema = ParameterEMA(model, config["ema_decay"]) if ema_enabled else None
    optimizer = optimizer_for(model, config)
    scaler = gpu_scaler(device, config["precision"])
    updates_per_epoch = (len(train) + config["batch_size"] - 1) // config["batch_size"]
    schedule = UpdateSchedule(optimizer, config, updates_per_epoch * config["epochs"])
    history, best_mae, best_epoch = [], float("inf"), 0
    run = provenance(config, device, manifest)
    run["data"]["training_images"], run["data"]["validation_images"] = len(train), len(val)
    run["weight_initialization"] = weight_provenance(config, config["pretrained"] and not resume)
    run["execution"] = "in_progress"
    run["updates_per_epoch"] = updates_per_epoch
    run["planned_updates"] = schedule.total_updates
    run["warmup_updates"] = schedule.warmup_updates
    last_path, best_path = output / "last.pt", output / "best.pt"
    data_sha = run["data"]["sha256"]
    checkpoint_index = []
    index_path = output / "checkpoint_index.json"
    if ema is not None or config.get("depth_screen"):
        boundaries = np.quantile([density(record) for record in train], (1 / 3, 2 / 3)).tolist()
    if ema is not None:
        run["ema"] = {"decay": ema.decay, "buffers": "live", "density_tertiles": boundaries}
        (output / "images").mkdir(exist_ok=resume)

    if resume:
        state = load_checkpoint(last_path, device)
        saved_config = DEFAULT_CONFIG | state["config"]
        compatible_fields = config.keys() if schedule.fixed else config.keys() - {"epochs"}
        legacy_constant = "scheduler" not in state and not schedule.fixed
        if (
            any(saved_config.get(k) != config[k] for k in compatible_fields)
            or saved_config.get("ema_decay") != config.get("ema_decay")
            or state["data_sha256"] != data_sha
            or (state["git_commit"] != run["git_commit"] and not legacy_constant)
        ):
            raise ValueError("resume requires matching commit, manifest and config for a fixed training horizon")
        history, best_mae, best_epoch = state["history"], state["best_mae"], state["best_epoch"]
        if end_epoch <= len(history):
            raise ValueError("resume must request an epoch beyond the last checkpoint")
        if ema is None and not best_path.is_file():
            raise ValueError("resume requires best.pt from the existing run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if legacy_constant:
            schedule.global_step = len(history) * updates_per_epoch
            if scaler.is_enabled():
                steps = [int(values["step"]) for values in optimizer.state.values() if "step" in values]
                schedule.global_step = max(steps, default=0)
        else:
            schedule.load_state_dict(state["scheduler"])
            if schedule.global_step != state["global_step"]:
                raise ValueError("checkpoint optimizer position and scheduler state differ")
        if ema is not None:
            if "ema" not in state:
                raise ValueError("EMA checkpoint is missing its shadow state")
            ema.load_state_dict(state["ema"], expected_step=schedule.global_step)
            checkpoint_index = json.loads(index_path.read_text(encoding="utf-8"))
            if (len(checkpoint_index) != len(history) or checkpoint_index[-1]["sha256"] != file_sha(last_path)):
                raise ValueError("EMA checkpoint provenance index differs")
        run["weight_initialization"] = state["weight_initialization"]
        if audited:
            initial_sha = state["initial_state_sha256"]
        run["resumed_from"] = {
            "epoch": len(history), "sha256": file_sha(last_path), "git_commit": state["git_commit"],
            "global_step": schedule.global_step,
        }

    if audited:
        run["initial_state_sha256"] = initial_sha
    write_json(output / "run_manifest.json", run)
    def checkpoint_state():
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": schedule.state_dict(), "global_step": schedule.global_step,
            "config": config, "data_sha256": data_sha, "history": history,
            "best_epoch": best_epoch, "best_mae": best_mae, "git_commit": run["git_commit"],
            "weight_initialization": run["weight_initialization"],
        }
        if audited:
            state["initial_state_sha256"] = initial_sha
        if ema is not None:
            state["ema"] = ema.state_dict()
        return state

    for epoch in range(len(history) + 1, end_epoch + 1):
        start = perf_counter()
        epoch_seed = config["seed"] + epoch
        seed_all(epoch_seed)
        loader = DataLoader(
            TrainingCrops(train, manifest.parent, config, epoch), batch_size=config["batch_size"],
            shuffle=True, num_workers=config["workers"], pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(epoch_seed),
        )
        if audited and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            with _phase(guard, "training"):
                details = _epoch(model, optimizer, scaler, loader, config, device, schedule, ema=ema, guard=guard)
            if guard is not None and details["optimizer_updates"] != updates_per_epoch:
                raise ValueError("missing optimizer updates in Stage 1 epoch")
            if ema is not None and ema.step != schedule.global_step:
                raise ValueError("EMA update position diverged from the optimizer")
            if expected is not None:
                actual = details["crop_order_sha256"], json_digest(details["learning_rates"])
                if actual != expected["trajectory"][epoch - 1]:
                    write_json(output / "mismatched_epoch.json", {"epoch": epoch, "training": details})
                    raise ValueError(f"observed crop order or LR differs at epoch {epoch}")
            details["train_seconds"] = perf_counter() - start
            if audited:
                details["mean_train_batch_ms"] = 1000 * details["train_seconds"] / len(loader)
                if device.type == "cuda":
                    details["peak_train_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    details["peak_train_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            stage_metrics = None
            validation_options = {}
            if "depth_readout_start_epoch" in config:
                validation_options["depth_diagnostic"] = epoch >= config["depth_readout_start_epoch"]
            if ema is not None:
                with preserve_training_context(model, device):
                    readouts, image_rows = {}, []
                    for policy, weights in (("live", nullcontext()), ("ema", ema.use_shadow())):
                        with _phase(guard, f"validation_{policy}"), weights:
                            metrics, rows = grouped_validation(
                                model, val, manifest.parent, config, device, staged=staged, boundaries=boundaries,
                                check=(lambda: guard.check("validation_image")) if guard is not None else (lambda: None),
                                **validation_options)
                            if model.channel_gate_deltas is not None:
                                details.setdefault("channel_gate_statistics", {})[policy] = model.channel_gate_statistics()
                        readouts[policy] = metrics
                        image_rows.extend({"policy": policy, **row} for row in rows)
                details["ema_validation"] = readouts["ema"]
                if config.get("depth_screen"):
                    details["depth_live_validation"] = readouts["live"]
                details["ema_step"] = ema.step
                stage_metrics = readouts["live"]
                image_path = output / "images" / f"{epoch:02d}.json"
                write_json(image_path, image_rows)
                details["image_metrics_sha256"] = file_sha(image_path)
            elif audited:
                if config.get("depth_screen"):
                    stage_metrics, rows = grouped_validation(
                        model, val, manifest.parent, config, device, staged=staged, boundaries=boundaries,
                        check=lambda: None, **validation_options)
                    details["depth_live_validation"] = stage_metrics
                else:
                    stage_metrics = evaluate_outputs(model, val, manifest.parent, config, device, staged=staged)
            if ema is None and model.channel_gate_deltas is not None:
                details["channel_gate_statistics"] = {"live": model.channel_gate_statistics()}
        except Exception as error:
            if guard is not None:
                partial = output / "interrupted.pt"
                checkpoint(partial, checkpoint_state() | {"interrupted_epoch": epoch})
                run.update(execution="stopped", reason=f"{type(error).__name__}: {error}",
                           completed_epochs=len(history), global_step=schedule.global_step,
                           interrupted_checkpoint={"path": str(partial), "sha256": file_sha(partial)})
                write_json(output / "run_manifest.json", run)
            raise
        details["epoch"] = epoch
        if stage_metrics is not None:
            if staged:
                details["early_validation"] = stage_metrics["early"]
            details["validation"] = stage_metrics["final"]
        else:
            gains = model.loop_gains()
            trace = LoopStatistics() if gains is not None else None
            details["validation"] = evaluate(model, val, manifest.parent, config, device, trace=trace)
            if trace is not None:
                details["loop_gains"] = gains.detach().tolist()
                details["validation_loop_statistics"] = trace.result()
        details["epoch_seconds"] = perf_counter() - start
        history.append(details)
        mae = details["validation"]["count"]["mae"]
        improved = ema is None and mae < best_mae
        if improved:
            best_mae, best_epoch = mae, epoch
        state = checkpoint_state()
        if improved:
            checkpoint(best_path, state)
        with _phase(guard, "checkpoint"):
            checkpoint(last_path, state)
        if ema is not None:
            checkpoint_index.append({"epoch": epoch, "step": ema.step, "sha256": file_sha(last_path)})
            write_json(index_path, checkpoint_index)
        write_json(output / "history.json", history)
        print(json.dumps(details, allow_nan=False), flush=True)
        if config["patience"] and epoch - best_epoch >= config["patience"]:
            break

    run["completed_epochs"] = len(history)
    run["best_epoch"] = best_epoch
    run["global_step"] = schedule.global_step
    run["execution"] = "paused" if len(history) == end_epoch < config["epochs"] else "completed"
    snapshots = (best_path, last_path) if ema is None else (last_path,)
    run["checkpoints"] = {p.name: {"path": str(p.resolve()), "sha256": file_sha(p)} for p in snapshots}
    write_json(output / "run_manifest.json", run)
    return run
