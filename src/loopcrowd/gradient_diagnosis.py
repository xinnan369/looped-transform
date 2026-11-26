"""Frozen two-call loss and call-gradient diagnostics."""

import torch
from torch import nn
from torch.func import functional_call

from .ema import preserve_training_context
from .inference import autocast
from .model import LoopCrowd
from .targets import point_loss

WEIGHT = 0.25
ZERO_NORM = 1e-12
TOLERANCES = {
    "fp32": {"outputs": (1e-6, 2e-5), "losses": (1e-6, 2e-5), "gradients": (2e-6, 2e-4)},
    "bf16": {"outputs": (0.002, 0.02), "losses": (0.0005, 0.01), "gradients": (0.001, 0.04)},
}
DECODER_PARTS = ("fuse.", "point_logits.", "point_offsets.")


class _PointDecoder(nn.Module):
    """Expose the existing decoder as a stateless-call target."""

    def __init__(self, model):
        super().__init__()
        self.fuse = model.fuse
        self.point_logits = model.point_logits
        self.point_offsets = model.point_offsets
        self.density = model.density
        self.slots = model.slots

    forward = LoopCrowd._decode


def _clone_parameters(parameters):
    return {name: value.detach().clone().requires_grad_(True) for name, value in parameters.items()}


def _select(parameters, prefixes):
    return {name: value for name, value in parameters.items() if name.startswith(prefixes)}


def _subtree(values, prefix):
    return {name.removeprefix(prefix): value for name, value in _select(values, prefix).items()}


def _expanded_forward(model, parameters, buffers, images):
    def run(module, prefix, *args):
        return functional_call(module, (_subtree(parameters, prefix), _subtree(buffers, prefix)), args, strict=True)

    # This expansion covers the frozen fixed-window two-call models. There is
    # one shared backbone and the second block consumes the undetached first result.
    decoder = _PointDecoder(model)
    p2 = run(model.p2, "p2.", images)
    tokens = run(model.project, "project.", run(model.p3, "p3.", p2)).permute(0, 2, 3, 1)
    outputs, leaves = {}, {}
    for index, stage in enumerate(("early", "final")):
        block_index = 0 if model.shared else index
        prefix = f"blocks.{block_index}."
        block = _clone_parameters(_subtree(parameters, prefix))
        head = _clone_parameters(_select(parameters, DECODER_PARTS))
        tokens = functional_call(model.blocks[block_index], (block, _subtree(buffers, prefix)), (tokens,), strict=True)
        outputs[stage] = functional_call(decoder, (head, _select(buffers, DECODER_PARTS)), (p2, tokens), strict=True)
        leaves[f"block_call_{index + 1}"] = block
        leaves[f"decoder_call_{index + 1}"] = head
    return outputs, leaves


def _native_groups(parameters, shared):
    blocks = {"block": _subtree(parameters, "blocks.0.")} if shared else {
        f"block_{index + 1}": _subtree(parameters, f"blocks.{index}.") for index in range(2)}
    return blocks | {"decoder": _select(parameters, DECODER_PARTS)}


def _losses(outputs, target):
    values = [value for stage in outputs.values() for value in stage.values()]
    if not all_finite(values):
        raise FloatingPointError("nonfinite probe output")
    early, final = (point_loss(outputs[stage], target) for stage in ("early", "final"))
    total = final + WEIGHT * early
    if not all_finite((early, final, total)):
        raise FloatingPointError("nonfinite probe loss")
    return {"early": early, "final": final, "total": total}


@torch.no_grad()
def all_finite(tensors):
    values = tuple(value.detach() for value in tensors)
    if not values:
        return True
    norms = torch._foreach_norm(values, ord=float("inf"))
    return bool(torch.isfinite(torch.stack(norms)).all())


def _vjp(loss, groups, *, retain_graph):
    tensors = [tensor for group in groups.values() for tensor in group.values()]
    if len({id(tensor) for tensor in tensors}) != len(tensors):
        raise ValueError("a probe parameter was listed more than once")
    values = torch.autograd.grad(loss, tensors, retain_graph=retain_graph, allow_unused=True)
    if not all_finite(value for value in values if value is not None):
        raise FloatingPointError("nonfinite probe VJP")
    gradients = iter(values)
    return {name: {key: next(gradients) for key in group} for name, group in groups.items()}


def _combine(a, b):
    if a.keys() != b.keys():
        raise ValueError("expanded gradient names differ")
    combined = {}
    for name, left in a.items():
        right = b[name]
        if left is None:
            combined[name] = right
        elif right is None:
            combined[name] = left
        else:
            combined[name] = left + right
    return combined


def _close(actual, expected, tolerance):
    atol, rtol = tolerance
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual.detach().float() - expected.detach().float()).abs().max())


def _gradient_equivalence(actual, expected, tolerance):
    if actual.keys() != expected.keys():
        raise ValueError("gradient mapping names differ")
    differences = []
    for name, before in expected.items():
        after = actual[name]
        if before is None or after is None:
            if not (before is None and after is None):
                raise ValueError(f"expanded autograd reachability differs for {name}")
            continue
        differences.append(_close(after, before, tolerance))
    return {"parameters_with_gradient": len(differences),
            "max_abs_delta": max(differences) if differences else None}


def gradient_pair(a, b, parameters, *, left_scale=1.0):
    """Summarize one exact parameter basis, retaining undefined cosines as null."""
    if a.keys() != parameters.keys() or b.keys() != parameters.keys() or not left_scale > 0:
        raise ValueError("gradient pairing requires aligned parameter names and a positive scale")

    def flatten(values):
        if all(value is None for value in values.values()):
            return None
        return torch.cat([(torch.zeros_like(parameters[name]) if value is None else value)
                          .detach().float().reshape(-1) for name, value in values.items()])

    left, right = flatten(a), flatten(b)
    result = {"left_scale": left_scale, "left_norm": None, "right_norm": None, "left_right_norm_ratio": None,
              "weighted_left_right_norm_ratio": None, "cosine": None, "weighted_left_projection_on_right": None,
              "right_projection_on_weighted_left": None, "conflict": None, "nonfinite": False}
    for label, values in (("left", left), ("right", right)):
        if values is not None:
            norm = torch.linalg.vector_norm(values)
            if not bool(torch.isfinite(norm)):
                return result | {"status": "nonfinite", "nonfinite": True}
            result[f"{label}_norm"] = float(norm)
    if left is None or right is None:
        return result | {"status": "no_gradient_path"}
    a_norm, b_norm = result["left_norm"], result["right_norm"]
    if min(a_norm, b_norm) <= ZERO_NORM:
        return result | {"status": "zero_or_near_zero_norm"}
    cosine = float(torch.dot(left / a_norm, right / b_norm).clamp(-1, 1))
    return result | {"status": "measured", "left_right_norm_ratio": a_norm / b_norm,
                     "weighted_left_right_norm_ratio": left_scale * a_norm / b_norm, "cosine": cosine,
                     "weighted_left_projection_on_right": left_scale * a_norm * cosine,
                     "right_projection_on_weighted_left": b_norm * cosine, "conflict": cosine < 0}


def diagnose_two_calls(model, images, target, *, precision="fp32", parameter_values=None):
    """Compare a native and expanded graph on generated CPU fixtures."""
    if images.device.type != "cpu":
        raise ValueError("the synthetic probe entry point requires CPU tensors")
    return _diagnose_two_calls(model, images, target, precision=precision, parameter_values=parameter_values)


def _diagnose_two_calls(model, images, target, *, precision, parameter_values=None):
    device = images.device
    if any(t.device != device for t in (*model.parameters(), *model.buffers(), *target.values())):
        raise ValueError("probe inputs and model must share one device")
    if (model.loops != 2 or model.shifted or model.embedding is not None or model.gain_logits is not None
            or model.channel_gate_deltas is not None
            or model.density is not None or model.gradient_checkpointing or model.freeze_backbone_bn):
        raise ValueError("gradient probe requires the frozen two-call LC-007B graph")
    if precision not in TOLERANCES:
        raise ValueError("unsupported probe precision")
    registered = dict(model.named_parameters())
    source = registered if parameter_values is None else parameter_values
    if source.keys() != registered.keys() or any(
        value.shape != registered[name].shape or value.dtype != registered[name].dtype or value.device != device
        for name, value in source.items()
    ):
        raise ValueError("live/EMA probe parameters must match the actual parameter schema")
    native = _clone_parameters(source)
    native_groups = _native_groups(native, model.shared)
    buffers = dict(model.named_buffers())
    tolerances = TOLERANCES[precision]
    with preserve_training_context(model, device):
        model.train()
        with autocast(device, precision):
            native_outputs = functional_call(
                model, (native, {key: value.clone() for key, value in buffers.items()}),
                (images,), {"return_stages": True}, strict=True)
            native_losses = _losses(native_outputs, target)
        native_gradients = {name: _vjp(native_losses[name], native_groups, retain_graph=name != "total")
                            for name in ("early", "final", "total")}
    with preserve_training_context(model, device):
        model.train()
        with autocast(device, precision):
            expanded_outputs, expanded_groups = _expanded_forward(
                model, source, {key: value.clone() for key, value in buffers.items()}, images)
            expanded_losses = _losses(expanded_outputs, target)
        output_checks = {f"{stage}_{field}": _close(value, native_outputs[stage][field], tolerances["outputs"])
                         for stage, output in expanded_outputs.items() for field, value in output.items()}
        loss_checks = {name: _close(value, native_losses[name], tolerances["losses"])
                       for name, value in expanded_losses.items()}
        expanded_gradients = {kind: _vjp(expanded_losses[kind], expanded_groups, retain_graph=kind == "final")
                              for kind in ("final", "total")}
    rows = [{"comparison": "early_vs_final_supervision", "group": group,
             **gradient_pair(native_gradients["early"][group], native_gradients["final"][group], parameters,
                             left_scale=WEIGHT)}
            for group, parameters in native_groups.items()]
    checks = {}
    for kind, expanded in expanded_gradients.items():
        native_gradient = native_gradients[kind]
        effective = {"decoder": _combine(expanded["decoder_call_1"], expanded["decoder_call_2"])}
        if model.shared:
            effective["block"] = _combine(expanded["block_call_1"], expanded["block_call_2"])
        else:
            effective.update(block_1=expanded["block_call_1"], block_2=expanded["block_call_2"])
        checks[kind] = {name: _gradient_equivalence(values, native_gradient[name], tolerances["gradients"])
                        for name, values in effective.items()}
        for group in ("block", "decoder"):
            relation = "descriptive_across_independent_blocks" if group == "block" and not model.shared else "same_parameter_reuse"
            first, second = f"{group}_call_1", f"{group}_call_2"
            rows.append({"comparison": f"{kind}_first_vs_second_call", "group": group, "relation": relation,
                         **gradient_pair(expanded[first], expanded[second], expanded_groups[first])})
    return {"outputs_max_abs": output_checks, "native_losses": {
                name: float(value.detach()) for name, value in native_losses.items()},
            "losses_max_abs": loss_checks, "gradient_equivalence": checks,
            "comparisons": rows}
