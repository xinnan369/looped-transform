"""Fixed-depth head-point models with a common CNN and point head."""

from pathlib import Path
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet18_Weights, resnet18

VARIANTS = {
    "cnn_only": (0, False),
    "shared_1": (1, True),
    "shared_2": (2, True),
    "shared_2_tied_gain": (2, True),
    "shared_2_stage_gain": (2, True),
    "unshared_2": (2, False),
    "shared_4": (4, True),
    "unshared_4": (4, False),
    "shifted_shared_2": (2, True),
    "shifted_unshared_2": (2, False),
    "shifted_shared_4": (4, True),
}
WEIGHTS = "IMAGENET1K_V1"
GAIN_STAGES = {"shared_2_tied_gain": 1, "shared_2_stage_gain": 2}


def pretrained_path():
    filename = ResNet18_Weights[WEIGHTS].url.rsplit("/", 1)[-1]
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def window_partition(tokens, size, shift=0):
    """Partition NHWC tokens into non-overlapping windows with a key padding mask."""
    batch, height, width, channels = tokens.shape
    padded_h = ((height + shift + size - 1) // size) * size
    padded_w = ((width + shift + size - 1) // size) * size
    # Shift the grid origin by padding, so opposite image edges never wrap into
    # the same window. The key mask excludes both the shifted and size padding.
    padded = F.pad(tokens, (0, 0, shift, padded_w - width - shift, shift, padded_h - height - shift))

    def partition(value):
        return (value.reshape(batch, padded_h // size, size, padded_w // size, size, -1)
                .permute(0, 1, 3, 2, 4, 5).reshape(-1, size * size, value.shape[-1]))

    row = torch.arange(padded_h, device=tokens.device)[:, None]
    col = torch.arange(padded_w, device=tokens.device)[None, :]
    rows = (row < shift) | (row >= height + shift)
    cols = (col < shift) | (col >= width + shift)
    padding = (rows | cols)[None, :, :, None].expand(batch, -1, -1, -1)
    return partition(padded), partition(padding).squeeze(-1)


def window_reverse(windows, shape, size, shift=0):
    batch, height, width, channels = shape
    rows = (height + shift + size - 1) // size
    cols = (width + shift + size - 1) // size
    padded = (windows.reshape(batch, rows, cols, size, size, channels)
              .permute(0, 1, 3, 2, 4, 5).reshape(batch, rows * size, cols * size, channels))
    return padded[:, shift:shift + height, shift:shift + width]


class WindowBlock(nn.Module):
    def __init__(self, width=192, heads=6, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, tokens, shift=0, *, gains=None, trace=None):
        before = tokens
        windows, padding = window_partition(self.norm1(tokens), self.window_size, shift)
        attended, _ = self.attn(windows, windows, windows, key_padding_mask=padding, need_weights=False)
        attention = window_reverse(attended, tokens.shape, self.window_size, shift)
        if gains is not None:
            attention = gains[0] * attention
        tokens = tokens + attention
        feedforward = self.ffn(self.norm2(tokens))
        if gains is not None:
            feedforward = gains[1] * feedforward
        tokens = tokens + feedforward
        if trace is not None:
            trace(before, tokens, attention, feedforward)
        return tokens


class LoopCrowd(nn.Module):
    stride = 4

    def __init__(
        self, variant, slots=2, pretrained=False, density=False,
        loop_embedding=False, gradient_checkpointing=False, freeze_backbone_bn=False,
    ):
        super().__init__()
        self.loops, self.shared = VARIANTS[variant]
        self.shifted = variant.startswith("shifted_")
        self.slots = slots
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_backbone_bn = freeze_backbone_bn

        # Initialize the common modules first so each control starts identically at the same seed.
        cnn = resnet18(weights=None)
        if pretrained:
            path = pretrained_path()
            if not path.is_file():
                raise FileNotFoundError(f"pretrained ResNet18 checkpoint must exist locally: {path}")
            cnn.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.p2 = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1)
        self.p3 = cnn.layer2
        self.project = nn.Conv2d(128, 192, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(64 + 192, 192, 3, padding=1, bias=False),
            nn.GroupNorm(24, 192), nn.ReLU(),
        )
        self.point_logits = nn.Conv2d(192, slots, 1)
        nn.init.constant_(self.point_logits.bias, -4.59511985)
        self.point_offsets = nn.Conv2d(192, slots * 2, 1)
        self.density = nn.Conv2d(192, 1, 1) if density else None
        self.embedding = nn.Parameter(torch.zeros(192)) if loop_embedding else None
        self.blocks = nn.ModuleList(WindowBlock() for _ in range(1 if self.shared else self.loops))
        stages = GAIN_STAGES.get(variant)
        self.gain_logits = nn.Parameter(torch.zeros(stages, 2)) if stages else None
        if self.freeze_backbone_bn:
            self.train()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for layer in (*self.p2.modules(), *self.p3.modules()):
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def loop_gains(self):
        if self.gain_logits is None:
            return None
        return (2 * self.gain_logits.sigmoid()).expand(self.loops, 2)

    def forward_stages(self, images):
        """Decode the first and second shared calls with the same point head."""
        return self(images, return_stages=True)

    def forward(self, images, *, loop_count=None, trace=None, return_stages=False):
        calls = self.loops if loop_count is None else loop_count
        if type(calls) is not int or not 0 <= calls <= self.loops:
            raise ValueError("loop_count must be an integer inside the model's fixed depth")
        if return_stages and (self.loops != 2 or calls != 2 or not self.shared):
            raise ValueError("stage predictions require both calls of a two-call shared model")
        p2 = self.p2(images)
        tokens = self.project(self.p3(p2)).permute(0, 2, 3, 1)
        gains = self.loop_gains()
        for index in range(calls):
            block = self.blocks[0 if self.shared else index]
            if self.embedding is not None:
                tokens = tokens + self.embedding
            shift = block.window_size // 2 if self.shifted and index % 2 else 0
            block_args = {}
            if gains is not None:
                block_args["gains"] = gains[index]
            if trace is not None:
                block_args["trace"] = partial(trace, index)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, shift, use_reentrant=False, **block_args)
            else:
                tokens = block(tokens, shift, **block_args)
            if return_stages and index == 0:
                early = self._decode(p2, tokens)
        final = self._decode(p2, tokens)
        return {"early": early, "final": final} if return_stages else final

    def _decode(self, p2, tokens):
        p3 = F.interpolate(tokens.permute(0, 3, 1, 2), size=p2.shape[-2:], mode="nearest")
        features = self.fuse(torch.cat((p2, p3), dim=1))
        batch, _, height, width = features.shape
        result = {
            "logits": self.point_logits(features),
            "offsets": self.point_offsets(features).reshape(batch, self.slots, 2, height, width).sigmoid(),
        }
        if self.density is not None:
            result["density"] = F.softplus(self.density(features))
        return result
