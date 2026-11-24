"""Train, validate and resume fixed-loop comparisons."""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrainingCrops, load_rgb
from .experiment import (
    checkpoint, file_sha, load_checkpoint, make_model, provenance, seed_all, weight_provenance, write_json,
)
from .inference import autocast, predict_image
from .manifest import read_manifest
from .metrics import SCORE_THRESHOLDS, count_metrics, point_nap, score_threshold_sweep
from .targets import point_loss


def gpu_scaler(device, precision):
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")


def optimizer_for(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


def train_step(model, optimizer, scaler, images, target, config, device, diagnostics=None):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device, config["precision"]):
        loss = point_loss(model(images), target, config["density_weight"])
    scaler.scale(loss).backward()
    if diagnostics is not None:
        scaler.unscale_(optimizer)
        norms = [torch.linalg.vector_norm(p.grad.detach().float())
                 for p in model.parameters() if p.grad is not None]
        diagnostics["gradient_l2_last_step"] = float(torch.linalg.vector_norm(torch.stack(norms)))
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


def evaluate(model, records, manifest_dir, config, device, all_predictions=None, sweep=False):
    model.eval()
    samples, count_samples, expected, observed = [], [], [], []
    summary = {}
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
    return result


def _epoch(model, optimizer, scaler, loader, config, device):
    model.train()
    losses = torch.zeros((), device=device)
    heads = overflow = images_seen = 0
    diagnostics = {}
    for step, (images, target) in enumerate(loader, 1):
        images_seen += len(images)
        heads += int(target["heads"].sum())
        overflow += int(target["overflow"].sum())
        gpu_target = {key: value.to(device, non_blocking=True) for key, value in target.items()}
        losses += train_step(model, optimizer, scaler, images.to(device, non_blocking=True),
                             gpu_target, config, device, diagnostics if step == len(loader) else None)
    mean_loss = float(losses / len(loader))
    if not np.isfinite(mean_loss) or not np.isfinite(diagnostics["gradient_l2_last_step"]):
        raise RuntimeError("training loss or observed gradient norm became non-finite")
    return {"loss": mean_loss, "crops": images_seen, "sampled_gt": heads, "overflow_gt": overflow,
            "overflow_rate": overflow / heads if heads else None,
            "learning_rate": optimizer.param_groups[0]["lr"], **diagnostics}


def fit(config, manifest, output, device, resume=False):
    records = read_manifest(manifest)
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    if not train or not val or len({record.dataset for record in train + val}) != 1:
        raise ValueError("fit requires one dataset with nonempty train and validation splits")
    if output.exists() and not resume:
        raise ValueError(f"new run requires a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=resume)
    seed_all(config["seed"])
    model = make_model(config, initialize=not resume).to(device)
    optimizer = optimizer_for(model, config)
    scaler = gpu_scaler(device, config["precision"])
    history, best_mae, best_epoch = [], float("inf"), 0
    run = provenance(config, device, manifest)
    run["data"]["training_images"], run["data"]["validation_images"] = len(train), len(val)
    run["weight_initialization"] = weight_provenance(config, config["pretrained"] and not resume)
    run["execution"] = "in_progress"
    last_path, best_path = output / "last.pt", output / "best.pt"
    data_sha = run["data"]["sha256"]

    if resume:
        state = load_checkpoint(last_path, device)
        if (
            {k: v for k, v in state["config"].items() if k != "epochs"}
            != {k: v for k, v in config.items() if k != "epochs"}
            or state["data_sha256"] != data_sha
            or state["git_commit"] != run["git_commit"]
        ):
            raise ValueError("resume requires the same commit, manifest and config (epochs may increase)")
        history, best_mae, best_epoch = state["history"], state["best_mae"], state["best_epoch"]
        if config["epochs"] <= len(history):
            raise ValueError("resume must request an epoch beyond the last checkpoint")
        if not best_path.is_file():
            raise ValueError("resume requires best.pt from the existing run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        run["weight_initialization"] = state["weight_initialization"]
        run["resumed_from"] = {"epoch": len(history), "sha256": file_sha(last_path)}

    write_json(output / "run_manifest.json", run)
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        epoch_seed = config["seed"] + epoch
        seed_all(epoch_seed)
        loader = DataLoader(
            TrainingCrops(train, manifest.parent, config, epoch), batch_size=config["batch_size"],
            shuffle=True, num_workers=config["workers"], pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(epoch_seed),
        )
        details = _epoch(model, optimizer, scaler, loader, config, device)
        details["epoch"] = epoch
        details["validation"] = evaluate(model, val, manifest.parent, config, device)
        history.append(details)
        mae = details["validation"]["count"]["mae"]
        improved = mae < best_mae
        if improved:
            best_mae, best_epoch = mae, epoch
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
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
    run["execution"] = "completed"
    run["checkpoints"] = {p.name: {"path": str(p.resolve()), "sha256": file_sha(p)}
                          for p in (best_path, last_path)}
    write_json(output / "run_manifest.json", run)
    return run
