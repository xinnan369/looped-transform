"""Evaluate count and pooled point nAP across fixed internal-val density groups."""

import numpy as np

from .accuracy_diagnosis import density, density_bin
from .calibration import DENSITY_GROUPS, count_match_metrics, rank_metrics, summarize_counts
from .data import load_rgb
from .inference import predict_image, predict_image_depth, predict_image_stages
from .metrics import pooled_point_nap


def summarize(rows, ranked):
    if not rows:
        return {"images": 0}
    counts = summarize_counts(rows, "matches")
    scores, assignments = zip(*ranked)
    nap = pooled_point_nap(scores, assignments, counts["gt"])
    if not all(np.isfinite(value) for value in (counts["mae"], counts["rmse"], nap["nap"]) if value is not None):
        raise ValueError("nonfinite validation metrics")
    return {
        "images": counts["images"], "count": {key: counts[key] for key in ("mae", "rmse")},
        "expected_heads": counts["gt"], "predicted_heads_on_labeled": counts["predicted"],
        "signed_count_error": counts["predicted"] - counts["gt"], "nap_local_v1": nap,
        **{key: value for key, value in counts.items() if key.startswith("local_")},
    }


def grouped_validation(model, records, directory, config, device, *, staged, boundaries, check, depth_diagnostic=True):
    model.eval()
    stages = ("early", "final") if staged else ("final",)
    depth = config.get("depth_screen", False)
    if depth:
        stages = tuple(model.depth_readout_stages(depth_diagnostic).values())
    state_stats = {stage: [] for stage in stages}
    gain_stats = []
    rows, ranked = ({stage: [] for stage in stages} for _ in range(2))
    threshold = config["count_threshold"]
    for record in records:
        if record.split != "val":
            raise ValueError("grouped evaluation requires internal-validation images")
        check()
        with load_rgb(record, directory) as image:
            if depth:
                gains = [] if config.get("refinement") else None
                outputs, norms = predict_image_depth(model, image, config, device, diagnostic=depth_diagnostic,
                                                     refinement_gains=gains)
                if gains is not None:
                    for gain, tile in zip(gains, outputs["final"].tiles, strict=True):
                        left, top, right, bottom = tile["image_xyxy"]
                        valid = gain[:(bottom - top + 7) // 8, :(right - left + 7) // 8]
                        if not np.isfinite(valid).all():
                            raise ValueError("nonfinite confidence refinement scale")
                        gain_stats.append((float(valid.min()), float(valid.mean()), float(valid.max())))
            else:
                outputs = (predict_image_stages(model, image, config, device, require_finite=True) if staged else
                           {"final": predict_image(model, image, config, device, require_finite=True)})
        group = density_bin(density(record), boundaries)
        for index, (stage, predictions) in enumerate(outputs.items()):
            active = predictions.count_candidates()
            active = active[active[:, 2] > threshold]
            matches, = count_match_metrics(record.points, active, (threshold,))
            rows[stage].append({"image_id": record.image_id, "gt": len(record.points),
                                "density_group": group, "matches": matches})
            ranked[stage].append(rank_metrics(record.points, predictions))
            if depth:
                state_stats[stage].append(norms[:, index].mean(axis=0, dtype=np.float64))
    check()
    results = {stage: summarize(rows[stage], ranked[stage]) | {
        "nap_score_floor": config["localization_min_score"], "local_match_delta": 0.50,
    } for stage in stages}
    image_rows = []
    for stage in stages:
        results[stage]["density_groups"] = {}
        if depth:
            state, update = np.sqrt(np.mean(state_stats[stage], axis=0))
            results[stage]["state_update"] = {"state_rms": float(state), "update_rms": float(update),
                                              "relative_rms": float(update / state) if state else None,
                                              "nonfinite": 0}
        for group in DENSITY_GROUPS:
            index = [i for i, row in enumerate(rows[stage]) if row["density_group"] == group]
            results[stage]["density_groups"][group] = summarize(
                [rows[stage][i] for i in index], [ranked[stage][i] for i in index])
        image_rows.extend({key: val for key, val in row.items() if key != "matches"} | row["matches"] | {
            "stage": stage,
        } for row in rows[stage])
    if gain_stats:
        low, means, high = zip(*gain_stats)
        results["final"]["refinement_gain"] = {"min": min(low), "mean_tile_mean": float(np.mean(means)), "max": max(high)}
    return results, image_rows
