"""Evaluate count and pooled point nAP across fixed internal-val density groups."""

import numpy as np

from .accuracy_diagnosis import density, density_bin
from .calibration import DENSITY_GROUPS, count_match_metrics, rank_metrics, summarize_counts
from .data import load_rgb
from .inference import predict_image, predict_image_depth, predict_image_stages
from .metrics import pooled_point_nap


class GainStatistics:
    """Pooled valid tile-cell moments for the actual and paired gain maps."""

    def __init__(self):
        self.cells = self.changed = 0
        self.minimum, self.maximum = 1.25, .75
        self.total = self.squares = self.cross = 0.

    def add(self, gain, width, height):
        actual = gain[:(height + 7) // 8, :(width + 7) // 8].astype(np.float64).ravel()
        if not len(actual) or not np.isfinite(actual).all() or actual.min() < .75 or actual.max() > 1.25:
            raise ValueError("invalid confidence refinement gain")
        # For a placebo map the inverse half-roll recovers the aligned values.
        # A circular shift has the same changed count and covariance as its inverse.
        values = actual - 1
        shifted = values.size // 2
        permuted = np.roll(values, shifted)
        self.cells += values.size
        self.changed += np.count_nonzero(values != permuted)
        self.minimum = min(self.minimum, float(actual.min()))
        self.maximum = max(self.maximum, float(actual.max()))
        self.total += float(values.sum())
        self.squares += float(np.einsum("i,i->", values, values))
        self.cross += float(np.einsum("i,i->", values, permuted))

    def result(self):
        if not self.cells:
            return None
        mean = self.total / self.cells
        variance = max(0., self.squares / self.cells - mean**2)
        # Repeated constant float32 values can leave a rounding residual in moments.
        correlation = ((self.cross / self.cells - mean**2) / variance
                       if self.minimum != self.maximum and variance > 0 else None)
        return {"min": self.minimum, "max": self.maximum, "mean": 1 + mean,
                "sd": float(np.sqrt(variance)) if self.minimum != self.maximum else 0.,
                "valid_tokens": int(self.cells), "changed_tokens": int(self.changed),
                "changed_fraction": float(self.changed / self.cells),
                "aligned_permuted_pearson": float(np.clip(correlation, -1, 1)) if correlation is not None else None}


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
    gain_stats = {group: GainStatistics() for group in ("all", *DENSITY_GROUPS)} if config.get("refinement") else {}
    rows, ranked = ({stage: [] for stage in stages} for _ in range(2))
    threshold = config["count_threshold"]
    for record in records:
        if record.split != "val":
            raise ValueError("grouped evaluation requires internal-validation images")
        check()
        group = density_bin(density(record), boundaries)
        with load_rgb(record, directory) as image:
            if depth:
                gains = [] if gain_stats else None
                outputs, norms = predict_image_depth(model, image, config, device, diagnostic=depth_diagnostic,
                                                     refinement_gains=gains)
                if gains is not None:
                    for gain, tile in zip(gains, outputs["final"].tiles, strict=True):
                        left, top, right, bottom = tile["image_xyxy"]
                        for bucket in ("all", group):
                            gain_stats[bucket].add(gain, right - left, bottom - top)
            else:
                outputs = (predict_image_stages(model, image, config, device, require_finite=True) if staged else
                           {"final": predict_image(model, image, config, device, require_finite=True)})
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
        results["final"]["refinement_gain"] = {group: stats.result() for group, stats in gain_stats.items()}
    return results, image_rows
