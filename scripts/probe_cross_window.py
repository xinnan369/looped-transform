"""Synthetic token-only reachability, with the CNN and decoder outside the graph."""

import torch

from loopcrowd.model import WindowBlock

TOLERANCE = {"fp32": 1e-6, "bf16": 0.001}
SCHEDULES = {"fixed": (0, 0, 0, 0), "shifted": (0, 4, 0, 4)}


def reachable_masks(height, width, origin, schedule):
    """Propagate possible communication on the real grid, without padding nodes."""
    row = torch.arange(height)[:, None]
    col = torch.arange(width)[None, :]
    reach = torch.zeros(height, width, dtype=torch.bool)
    reach[origin] = True
    for shift in schedule:
        # Padding changes the window origin on the real grid. IDs need only be unique.
        windows = ((row + shift) // 8) * (width + 8) + (col + shift) // 8
        reach = torch.isin(windows, windows[reach].unique())
        yield reach


def probe(precision):
    torch.manual_seed(42)
    block = WindowBlock()
    use_bf16 = precision == "bf16"
    tokens = torch.randn(2, 17, 19, 192)
    if use_bf16:
        tokens = tokens.bfloat16()
    origin = (7, 7)
    target_cells = ((7, 7), (8, 8), (15, 15), (16, 16))
    perturbed = tokens.clone()
    perturbed[0, origin[0], origin[1]] += torch.linspace(-1.5, 1.5, 192).to(tokens.dtype)
    batch = torch.cat((tokens, perturbed)).requires_grad_()
    evidence = {}
    for policy, schedule in SCHEDULES.items():
        states = []
        state = batch
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
            for shift in schedule:
                state = block(state, shift)
                states.append(state)
        results = []
        for call, (state, reach, target) in enumerate(
            zip(states, reachable_masks(17, 19, origin, schedule), target_cells, strict=True), 1
        ):
            delta = (state[0].float() - state[2].float()).abs().amax(-1)
            observed = delta > TOLERANCE[precision]
            y, x = target
            gradient, = torch.autograd.grad(state[0, y, x].float().square().mean(), batch,
                                            retain_graph=call < len(schedule))
            grad_at_source = gradient[0, origin[0], origin[1]].float().norm()
            if (not torch.isfinite(state).all() or not torch.isfinite(gradient).all()
                    or (observed & ~reach).any() or not torch.equal(state[1], state[3])
                    or bool(observed[target]) != bool(reach[target])
                    or bool(grad_at_source > 0) != bool(reach[target])):
                raise ValueError(f"synthetic propagation or isolation failed at {precision}/{policy}/{call}")
            results.append({"call": call, "shift": schedule[call - 1], "block_index": 0,
                            "possible_tokens": int(reach.sum()), "observed_tokens": int(observed.sum()),
                            "max_abs_delta": float(delta.detach().max()), "target": list(target),
                            "target_abs_delta": float(delta[target].detach()), "source_gradient_l2": float(grad_at_source),
                            "max_outside_oracle": float(delta[~reach].detach().max()) if (~reach).any() else 0.0})
        evidence[policy] = results
    return evidence
