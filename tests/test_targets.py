import numpy as np
import pytest
import torch

from loopcrowd.crops import crop_origin, crop_points
from loopcrowd.targets import point_loss, point_targets


def test_slots_overflow_and_partial_grid():
    points = [[9.99, 6.99], [3, 3], [0, 0], [1, 2], [4, 4]]
    target = point_targets(points, 12, 12, valid_width=10, valid_height=7)
    assert target["heads"].item() == 5
    assert target["overflow"].item() == 1
    assert target["density"].sum().item() == 5
    assert target["labels"].sum().item() == 4
    assert target["offsets"][1, :, 0, 0].tolist() == [0.25, 0.5]
    assert target["valid"].sum().item() == 6
    assert target["labels"][0, 1, 2].item() == 1


@pytest.mark.parametrize("point", [[10, 6], [0, 7], [-1, 1], [float("nan"), 0]])
def test_rejects_out_of_image_points(point):
    with pytest.raises(ValueError, match="within the valid image region"):
        point_targets([point], 12, 12, valid_width=10, valid_height=7)


def test_empty_points_and_positive_only_offset():
    target = {name: value[None] for name, value in point_targets([], 8, 8).items()}
    offsets = torch.rand(1, 2, 2, 2, 2, requires_grad=True)
    logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
    loss = point_loss({"offsets": offsets, "logits": logits}, target)
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert offsets.grad.abs().sum() == 0


def test_density_biased_is_repeatable_and_at_least_uniform():
    points = np.repeat([[510.5, 512.5], [1022, 1021]], [30, 2], axis=0)
    for seed in (42, 123, 2026):
        arguments = (seed, "fixture")
        uniform = crop_origin(points, 1200, 1200, 512, "random", *arguments)
        biased = crop_origin(points, 1200, 1200, 512, "density-biased", *arguments)
        assert biased == crop_origin(points, 1200, 1200, 512, "density-biased", *arguments)
        assert len(crop_points(points, *biased, 512)) >= len(crop_points(points, *uniform, 512))
