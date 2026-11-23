import pytest
import torch

from loopcrowd.model import LoopCrowd, VARIANTS, WindowBlock, window_partition, window_reverse


def test_partition_reverse_and_padding():
    value = torch.arange(2 * 9 * 13 * 3).reshape(2, 9, 13, 3).float()
    windows, mask = window_partition(value, 8)
    assert windows.shape == (8, 64, 3)
    assert int(mask.sum()) == 2 * (16 * 16 - 9 * 13)
    assert torch.equal(window_reverse(windows, value.shape, 8), value)
    assert not mask.all(dim=1).any()


@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_forward_backward(variant):
    model = LoopCrowd(variant)
    output = model(torch.randn(2, 3, 35, 41))
    assert output["logits"].shape == (2, 2, 9, 11)
    assert output["offsets"].shape == (2, 2, 2, 9, 11)
    (output["logits"].square().mean() + output["offsets"].mean()).backward()
    assert model.point_logits.weight.grad.isfinite().all()


def test_shared_parameter_identity_gradients_and_control_size():
    parameters = {}
    for variant in VARIANTS:
        torch.manual_seed(42)
        candidate = LoopCrowd(variant)
        parameters[variant] = sum(p.numel() for p in candidate.parameters())
        if variant == "cnn_only":
            reference = candidate.point_logits.weight.detach().clone()
        assert torch.equal(candidate.point_logits.weight, reference)
    assert parameters["shared_1"] == parameters["shared_2"] == parameters["shared_4"]
    assert parameters["unshared_4"] - parameters["cnn_only"] == 4 * (
        parameters["shared_4"] - parameters["cnn_only"])
    model = LoopCrowd("shared_4")
    block = model.blocks[0]
    weight = block.attn.in_proj_weight
    observed, direct_gradients, output_gradients = [], [], []

    def hook(module, arguments, output):
        observed.append(module.attn.in_proj_weight)
        # Detach previous passes to test this invocation's direct weight dependency.
        isolated = module.forward(arguments[0].detach())
        direct_gradients.append(torch.autograd.grad(isolated.square().mean(), weight)[0])
        output.register_hook(output_gradients.append)

    handle = block.register_forward_hook(hook)
    output = model(torch.randn(2, 3, 40, 40))
    handle.remove()
    output["logits"].square().mean().backward()
    assert len(model.blocks) == 1
    assert len(observed) == 4 and all(parameter is weight for parameter in observed)
    for gradients in (direct_gradients, output_gradients):
        assert len(gradients) == 4
        assert all(gradient.isfinite().all() and gradient.abs().sum() > 0 for gradient in gradients)
    assert weight.grad.abs().sum() > 0
    independent = LoopCrowd("unshared_4")
    assert len({id(block.attn.in_proj_weight) for block in independent.blocks}) == 4


def test_checkpointed_loop_and_density_backward():
    model = LoopCrowd("shared_2", density=True, gradient_checkpointing=True, loop_embedding=True)
    output = model(torch.ones(1, 3, 33, 37))
    output["density"].mean().backward()
    assert model.blocks[0].attn.in_proj_weight.grad.abs().sum() > 0
    assert model.embedding.grad.abs().sum() > 0


def test_shifted_partition_padding_and_next_pass_attention():
    torch.manual_seed(42)
    tokens = torch.randn(1, 16, 16, 192)
    windows, mask = window_partition(tokens, 8, shift=4)
    assert windows.shape == (9, 64, 192)
    assert int(mask.sum()) == 24 * 24 - 16 * 16
    assert not mask.all(dim=1).any()
    assert torch.equal(window_reverse(windows, tokens.shape, 8, shift=4), tokens)
    block = WindowBlock().eval()
    with torch.no_grad():
        first = block(tokens)
        changed = first.clone()
        changed[0, 5, 8] = torch.randn(192)
        fixed, altered_fixed = block(first), block(changed)
        shifted, altered_shifted = block(first, 4), block(changed, 4)
        assert torch.equal(fixed[0, 5, 7], altered_fixed[0, 5, 7])
        assert not torch.allclose(shifted[0, 5, 7], altered_shifted[0, 5, 7])
        opposite_edge = first.clone()
        opposite_edge[0, 15, 0] = torch.randn(192)
        opposite_edge[0, 0, 15] = torch.randn(192)
        assert torch.equal(shifted[0, 0, 0], block(opposite_edge, 4)[0, 0, 0])


def test_shifted_variant_alternates_with_one_parameter_object():
    model = LoopCrowd("shifted_shared_4")
    observed = []
    model.blocks[0].register_forward_pre_hook(
        lambda block, args: observed.append((block.attn.in_proj_weight, args[1]))
    )
    output = model(torch.randn(2, 3, 47, 61))
    output["logits"].square().mean().backward()
    assert [shift for _, shift in observed] == [0, 4, 0, 4]
    assert all(weight is model.blocks[0].attn.in_proj_weight for weight, _ in observed)
    assert model.blocks[0].attn.in_proj_weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in LoopCrowd("shared_4").parameters())
