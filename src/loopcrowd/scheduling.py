"""Learning rates indexed by completed optimizer updates."""

from math import cos, pi


class UpdateSchedule:
    def __init__(self, optimizer, config, total_updates):
        self.optimizer = optimizer
        self.kind = config["lr_schedule"]
        self.peak = config["lr"]
        self.minimum = config["min_lr"]
        self.total_updates = total_updates
        self.fixed = config["fixed_training_horizon"] or self.kind != "constant"
        self.warmup_updates = 0
        if self.kind == "warmup_cosine":
            if total_updates < 3:
                raise ValueError("warmup cosine requires at least three planned updates")
            self.warmup_updates = max(1, min(total_updates - 2, int(total_updates * config["warmup_fraction"])))
        self.global_step = 0

    def apply(self):
        """Set the LR for the next attempted update without advancing the clock."""
        if self.global_step >= self.total_updates:
            raise ValueError("optimizer update exceeds the planned training horizon")
        rate = self.peak
        if self.warmup_updates:
            if self.global_step < self.warmup_updates:
                progress = self.global_step / max(1, self.warmup_updates - 1)
                rate = self.minimum + (self.peak - self.minimum) * progress
            else:
                progress = (self.global_step - self.warmup_updates) / (self.total_updates - self.warmup_updates - 1)
                rate = self.minimum + (self.peak - self.minimum) * (1 + cos(pi * progress)) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def state_dict(self):
        return {
            "kind": self.kind, "peak": self.peak, "minimum": self.minimum,
            "total_updates": self.total_updates, "warmup_updates": self.warmup_updates,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for field in ("kind", "peak", "minimum", "warmup_updates"):
            if state[field] != expected[field]:
                raise ValueError(f"resume changed scheduler field {field}")
        if self.fixed and state["total_updates"] != self.total_updates:
            raise ValueError("resume changed the fixed optimizer update horizon")
        if not 0 <= state["global_step"] <= min(state["total_updates"], self.total_updates):
            raise ValueError("invalid saved optimizer update position")
        self.global_step = state["global_step"]
