"""Train, validate and resume fixed-loop comparisons."""

import json
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrainingCrops, load_rgb
from .experiment import (
    DEFAULT_CONFIG, checkpoint, file_sha, load_checkpoint, make_model, provenance, seed_all,
    weight_provenance, write_json,
)
from .inference import autocast, predict_image
from .manifest import read_manifest
from .metrics import SCORE_THRESHOLDS, count_metrics, point_nap, score_threshold_sweep
from .scheduling import UpdateSchedule
from .targets import point_loss


def gpu_scaler(device, precision):
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")


def optimizer_for(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


def train_step(model, optimizer, scaler, images, target, config, device, diagnostics=None):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device, config["precision"]):
        loss = point_loss(model(images), target, config["density_weight"], diagnostics)
    scaler.scale(loss).backward()
    if diagnostics is not None:
        scaler.unscale_(optimizer)
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        norms = torch._foreach_norm(grads)
        diagnostics["gradient_l2"] = torch.linalg.vector_norm(torch.stack(norms))
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


def evaluate(model, records, manifest_dir, config, device, all_predictions=None, sweep=False, per_image=False):
    model.eval()
    samples, count_samples, expected, observed = [], [], [], []
    summary = {}
    image_counts = []
    threshold = config["count_threshold"]
    if sweep and config["localization_min_score"] > min(SCORE_THRESHOLDS):
        raise ValueError("score floor is above the first validation sweep threshold")
    for record in records:
        with load_rgb(record, manifest_dir) as image:
            predictions = predict_image(model, image, config, device)
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


def _epoch(model, optimizer, scaler, loader, config, device, schedule):
    model.train()
    losses = torch.zeros((), device=device)
    class_losses = torch.zeros_like(losses)
    offsets = torch.zeros_like(losses)
    heads = overflow = images_seen = valid_slots = skipped = 0
    rates, gradient_norms = [], []
    scale = scaler.get_scale()
    for images, target in loader:
        images_seen += len(images)
        heads += int(target["heads"].sum())
        overflow += int(target["overflow"].sum())
        valid_slots += int(target["valid"].sum()) * config["slots"]
        gpu_target = {key: value.to(device, non_blocking=True) for key, value in target.items()}
        diagnostics = {}
        rate = schedule.apply()
        loss = train_step(model, optimizer, scaler, images.to(device, non_blocking=True),
                          gpu_target, config, device, diagnostics)
        next_scale = scaler.get_scale()
        if next_scale >= scale:
            schedule.global_step += 1
            rates.append(rate)
        else:
            skipped += 1
        scale = next_scale
        losses += loss
        class_losses += diagnostics["classification_loss"]
        offsets += diagnostics["offset_loss"]
        gradient_norms.append(diagnostics["gradient_l2"])
    mean_loss = float(losses / len(loader))
    gradients = torch.stack(gradient_norms).cpu().numpy()
    finite = gradients[np.isfinite(gradients)]
    if not np.isfinite(mean_loss) or len(finite) + skipped < len(gradients):
        raise RuntimeError("training loss or observed gradient norm became non-finite")
    positive_slots = heads - overflow
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
    }


def fit(config, manifest, output, device, resume=False, stop_after=None):
    config = DEFAULT_CONFIG | config
    records = read_manifest(manifest)
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
    seed_all(config["seed"])
    model = make_model(config, initialize=not resume).to(device)
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

    if resume:
        state = load_checkpoint(last_path, device)
        saved_config = DEFAULT_CONFIG | state["config"]
        compatible_fields = config.keys() if schedule.fixed else config.keys() - {"epochs"}
        legacy_constant = "scheduler" not in state and not schedule.fixed
        if (
            any(saved_config[k] != config[k] for k in compatible_fields)
            or state["data_sha256"] != data_sha
            or (state["git_commit"] != run["git_commit"] and not legacy_constant)
        ):
            raise ValueError("resume requires matching commit, manifest and config for a fixed training horizon")
        history, best_mae, best_epoch = state["history"], state["best_mae"], state["best_epoch"]
        if end_epoch <= len(history):
            raise ValueError("resume must request an epoch beyond the last checkpoint")
        if not best_path.is_file():
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
        run["weight_initialization"] = state["weight_initialization"]
        run["resumed_from"] = {
            "epoch": len(history), "sha256": file_sha(last_path), "git_commit": state["git_commit"],
            "global_step": schedule.global_step,
        }

    write_json(output / "run_manifest.json", run)
    for epoch in range(len(history) + 1, end_epoch + 1):
        start = perf_counter()
        epoch_seed = config["seed"] + epoch
        seed_all(epoch_seed)
        loader = DataLoader(
            TrainingCrops(train, manifest.parent, config, epoch), batch_size=config["batch_size"],
            shuffle=True, num_workers=config["workers"], pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(epoch_seed),
        )
        details = _epoch(model, optimizer, scaler, loader, config, device, schedule)
        details["train_seconds"] = perf_counter() - start
        details["epoch"] = epoch
        details["validation"] = evaluate(model, val, manifest.parent, config, device)
        details["epoch_seconds"] = perf_counter() - start
        history.append(details)
        mae = details["validation"]["count"]["mae"]
        improved = mae < best_mae
        if improved:
            best_mae, best_epoch = mae, epoch
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": schedule.state_dict(), "global_step": schedule.global_step,
            "config": config, "data_sha256": data_sha, "history": history,
            "best_epoch": best_epoch, "best_mae": best_mae, "git_commit": run["git_commit"],
            "weight_initialization": run["weight_initialization"],
        }
        if improved:
            checkpoint(best_path, state)
        checkpoint(last_path, state)
        write_json(output / "history.json", history)
        print(json.dumps(details, allow_nan=False), flush=True)
        if config["patience"] and epoch - best_epoch >= config["patience"]:
            break

    run["completed_epochs"] = len(history)
    run["best_epoch"] = best_epoch
    run["global_step"] = schedule.global_step
    run["execution"] = "paused" if len(history) == end_epoch < config["epochs"] else "completed"
    run["checkpoints"] = {p.name: {"path": str(p.resolve()), "sha256": file_sha(p)}
                          for p in (best_path, last_path)}
    write_json(output / "run_manifest.json", run)
    return run
