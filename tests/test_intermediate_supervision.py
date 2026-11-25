from copy import deepcopy

import numpy as np
import torch
from PIL import Image

from loopcrowd.experiment import DEFAULT_CONFIG, load_checkpoint, make_model, state_sha
from loopcrowd.inference import predict_image, predict_image_stages
from loopcrowd.manifest import Record, write_manifest
from loopcrowd.model import LoopCrowd
from loopcrowd.targets import point_loss, point_targets
from loopcrowd.training import fit, supervised_point_loss


def training_batch():
    target = point_targets(np.array([[8.25, 13.75], [22., 25.]]), 40, 40, 2)
    return torch.randn(2, 3, 40, 40), {key: torch.stack((value, value)) for key, value in target.items()}


def test_stages_reuse_one_block_and_existing_decoder_parameters():
    model = LoopCrowd("shared_2").eval()
    before = state_sha(model)
    seen = {}
    watched = {
        "block": model.blocks[0], "fuse": model.fuse,
        "logits": model.point_logits, "offsets": model.point_offsets,
    }
    handles = [module.register_forward_pre_hook(
        lambda module, args, key=key: seen.setdefault(key, []).append(tuple(map(id, module.parameters())))
    ) for key, module in watched.items()]
    images, target = training_batch()
    with torch.inference_mode():
        stages = model.forward_stages(images)
        assert set(stages) == {"early", "final"}
        for outputs in stages.values():
            assert outputs["logits"].shape == target["labels"].shape
            assert outputs["offsets"].shape == target["offsets"].shape
    assert all(len(calls) == 2 and calls[0] == calls[1] for calls in seen.values())
    seen.clear()
    with torch.inference_mode():
        ordinary = model(images)
    assert len(seen["block"]) == 2 and len(seen["logits"]) == len(seen["offsets"]) == len(seen["fuse"]) == 1
    for field, value in ordinary.items():
        torch.testing.assert_close(stages["final"][field], value, rtol=0, atol=0)
    for handle in handles:
        handle.remove()
    assert state_sha(model) == before
    assert sum(p.numel() for p in model.parameters()) == 1_596_614


def test_zero_weight_matches_legacy_and_aux_backpropagates_into_shared_core():
    original = LoopCrowd("shared_2")
    zero, aux = deepcopy(original), deepcopy(original)
    images, target = training_batch()
    legacy = point_loss(original(images), target)
    parts = {}
    staged = supervised_point_loss(zero, images, target, DEFAULT_CONFIG | {"early_loss_weight": 0.0}, parts)
    weighted = supervised_point_loss(aux, images, target, DEFAULT_CONFIG | {"early_loss_weight": 0.25})
    torch.testing.assert_close(staged, legacy, rtol=0, atol=0)
    torch.testing.assert_close(weighted, staged + 0.25 * parts["early_loss"], rtol=0, atol=0)
    for loss in (legacy, staged, weighted):
        loss.backward()
    for a, b, c in zip(original.parameters(), zero.parameters(), aux.parameters(), strict=True):
        assert all(torch.isfinite(p.grad).all() for p in (a, b, c))
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
    for key in ("blocks.0.attn.in_proj_weight", "point_logits.weight", "point_offsets.weight"):
        g0, g1 = (dict(model.named_parameters())[key].grad for model in (zero, aux))
        assert torch.linalg.vector_norm(g1 - g0) > 1e-6


def test_normal_tiled_prediction_and_old_state_load(tmp_path):
    original = LoopCrowd("shared_2").eval()
    old_path = tmp_path / "old.pt"
    old_config = {k: v for k, v in DEFAULT_CONFIG.items() if k != "early_loss_weight"} | {
        "variant": "shared_2", "tile_size": 32, "tile_overlap": 8,
    }
    torch.save({"config": old_config, "model": original.state_dict()}, old_path)
    state = load_checkpoint(old_path, torch.device("cpu"))
    restored = make_model(state["config"], initialize=False).eval()
    restored.load_state_dict(state["model"])
    with Image.new("RGB", (41, 35), (150, 33, 84)) as image:
        default = predict_image(restored, image, old_config, torch.device("cpu"))
        staged = predict_image_stages(restored, image, old_config, torch.device("cpu"))
    assert set(staged) == {"early", "final"}
    for name in ("candidates", "tile_ids", "owned"):
        np.testing.assert_array_equal(getattr(default, name), getattr(staged["final"], name))


def test_stage_training_smoke_saves_and_resumes_loss_and_crop_record(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    image = tmp_path / "fixture.png"
    Image.new("RGB", (33, 37), (90, 80, 20)).save(image)
    records = [Record(str(i), "fixture", split, image.name, 33, 37, [[10., 12.], [24., 20.]])
               for i, split in enumerate(("train", "train", "val"))]
    write_manifest(manifest, records)
    config = DEFAULT_CONFIG | {
        "variant": "shared_2", "epochs": 2, "tile_size": 32, "tile_overlap": 8,
        "fixed_training_horizon": True, "early_loss_weight": 0.25,
    }
    output, device = tmp_path / "train", torch.device("cpu")
    paused = fit(config, manifest, output, device, stop_after=1)
    assert paused["execution"] == "paused" and paused["global_step"] == 1
    complete = fit(config, manifest, output, device, resume=True)
    assert complete["execution"] == "completed" and complete["global_step"] == 2
    saved = load_checkpoint(output / "last.pt", device)
    assert paused["initial_state_sha256"] == saved["initial_state_sha256"] == complete["initial_state_sha256"]
    for row in saved["history"]:
        assert row["loss"] == row["final_loss"] + 0.25 * row["early_loss"] or np.isclose(
            row["loss"], row["final_loss"] + 0.25 * row["early_loss"]
        )
        assert len(row["crop_order_sha256"]) == 64 and row["optimizer_updates"] == 1
        for stage in ("early_validation", "validation"):
            assert row[stage]["local_tp"] + row[stage]["local_fn"] == row[stage]["expected_heads"]
        assert 0 < row["early_classification_loss"] and 0 < row["early_offset_loss"]
