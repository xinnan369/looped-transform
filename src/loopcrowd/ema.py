"""FP32 parameter EMA with explicit checkpoint and live-parameter swap semantics."""

import math
import random
from contextlib import contextmanager

import numpy as np
import torch


def validate_decay(decay):
    if type(decay) not in (float, int) or not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("EMA decay must be finite and strictly between zero and one")


class ParameterEMA:
    def __init__(self, model, decay):
        validate_decay(decay)
        self.decay = float(decay)
        self.step = 0
        self.parameters = dict(model.named_parameters())
        self.shadow = {name: parameter.detach().to(dtype=torch.float32, copy=True)
                       for name, parameter in self.parameters.items()}
        self.require_finite()

    @torch.no_grad()
    def update(self, *, updated):
        if updated:
            shadow = tuple(self.shadow.values())
            live = tuple(parameter.detach().float() for parameter in self.parameters.values())
            torch._foreach_mul_(shadow, self.decay)
            torch._foreach_add_(shadow, live, alpha=1 - self.decay)
            self.step += 1

    @torch.no_grad()
    def require_finite(self, *, check_live=False):
        tensors = tuple(self.shadow.values())
        if check_live:
            tensors += tuple(self.parameters.values())
        norms = torch._foreach_norm(tensors, ord=float("inf"))
        if not torch.isfinite(torch.stack(norms)).all():
            raise ValueError("nonfinite EMA shadow or live parameter")

    def state_dict(self):
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow}

    @torch.no_grad()
    def load_state_dict(self, state, *, expected_step):
        if state["decay"] != self.decay or type(state["step"]) is not int or state["step"] != expected_step:
            raise ValueError("EMA checkpoint decay or update position differs")
        if state["shadow"].keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint parameter names differ")
        values = []
        for name, target in self.shadow.items():
            source = state["shadow"][name]
            if source.shape != target.shape or source.dtype != torch.float32:
                raise ValueError(f"EMA checkpoint shadow tensor differs for {name}")
            values.append(source.to(target.device))
        if not torch.isfinite(torch.stack(torch._foreach_norm(values, ord=float("inf")))).all():
            raise ValueError("EMA checkpoint contains nonfinite shadow values")
        torch._foreach_copy_(tuple(self.shadow.values()), values)
        self.step = state["step"]

    @contextmanager
    def use_shadow(self):
        """Swap only unique parameters and restore live tensors even on validation failure."""
        parameters = tuple(self.parameters.values())
        original = [parameter.detach().clone() for parameter in parameters]
        try:
            with torch.no_grad():
                torch._foreach_copy_(parameters, tuple(self.shadow.values()))
            yield
        finally:
            with torch.no_grad():
                torch._foreach_copy_(parameters, original)


@contextmanager
def preserve_training_context(model, device):
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
