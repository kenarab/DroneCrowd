"""Strict CPU/CUDA inference adapter for official MPCount STA deterministic weights.

Derived from Shimmer93/MPCount, ``models/models.py`` at commit
6eb06772bcf7dfb771c43a14d67146fce767f103. The adapter keeps the official
``DGModel_final`` parameter names and forward semantics, while constructing
only VGG16-BN feature layers (no ImageNet download or unused classifier).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F


SOURCE_COMMIT = "6eb06772bcf7dfb771c43a14d67146fce767f103"
SOURCE_REPOSITORY = "https://github.com/Shimmer93/MPCount"


def _vgg16_bn_features() -> nn.Sequential:
    """Torchvision VGG16-BN features with matching Sequential state keys."""
    config = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"]
    layers: list[nn.Module] = []
    input_channels = 3
    for channels in config:
        if channels == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            output_channels = int(channels)
            layers.extend((nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)))
            input_channels = output_channels
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """MPCount ConvBlock, with the upstream defaults preserved."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FixedUpsample(nn.Module):
    """Learned depthwise upsampling kernel used by deterministic MPCount."""
    def __init__(self, channel: int, scale_factor: int):
        super().__init__()
        if not isinstance(scale_factor, int) or scale_factor <= 1 or scale_factor % 2:
            raise ValueError("MPCount fixed upsample scale must be an even integer > 1")
        self.scale_factor = scale_factor
        kernel_size = scale_factor + 1
        self.weight = nn.Parameter(torch.empty(channel, 1, kernel_size, kernel_size))
        with torch.no_grad():
            self.weight.fill_(1 / (kernel_size * kernel_size))

    def forward(self, x):
        if x is None:
            return None
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return F.conv2d(x, self.weight, padding=self.scale_factor // 2,
                        groups=self.weight.shape[0])


class Upsample(nn.Module):
    def __init__(self, channel: int, scale_factor: int):
        super().__init__()
        self.deterministic = True
        self.upsample = FixedUpsample(channel, scale_factor)

    def forward(self, x):
        return self.upsample(x)


class DGModel_final(nn.Module):
    """Official MPCount final density/PAC model with checkpoint-identical keys."""
    def __init__(self, mem_size=1024, mem_dim=256, cls_thrs=0.5,
                 den_dropout=0.5, cls_dropout=0.3):
        super().__init__()
        features = _vgg16_bn_features()
        self.enc1 = nn.Sequential(*list(features.children())[:23])
        self.enc2 = nn.Sequential(*list(features.children())[23:33])
        self.enc3 = nn.Sequential(*list(features.children())[33:43])
        self.dec3 = nn.Sequential(ConvBlock(512, 1024, bn=True),
                                  ConvBlock(1024, 512, bn=True))
        self.dec2 = nn.Sequential(ConvBlock(1024, 512, bn=True),
                                  ConvBlock(512, 256, bn=True))
        self.dec1 = nn.Sequential(ConvBlock(512, 256, bn=True),
                                  ConvBlock(256, 128, bn=True))
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.mem = nn.Parameter(torch.FloatTensor(1, mem_dim, mem_size).normal_(0.0, 1.0))
        self.den_dec = nn.Sequential(ConvBlock(512+256+128, mem_dim,
                                               kernel_size=1, padding=0, bn=True))
        self.den_head = nn.Sequential(ConvBlock(mem_dim, 1, kernel_size=1, padding=0))
        self.cls_thrs = cls_thrs
        self.cls_dropout = cls_dropout
        self.cls_head = nn.Sequential(ConvBlock(512, 256, bn=True),
                                      nn.Dropout2d(p=cls_dropout),
                                      ConvBlock(256, 1, kernel_size=1, padding=0,
                                                relu=False), nn.Sigmoid())
        self.upsample1 = Upsample(512, 2)
        self.upsample2 = Upsample(256, 2)
        self.upsample3 = Upsample(256, 2)
        self.upsample4 = Upsample(512, 4)
        self.upsample_d = Upsample(1, 4)

    def forward_fe(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        y3 = self.dec3(x3)
        x = self.upsample1(y3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        y2 = x
        x = self.upsample2(x)
        y1 = self.dec1(torch.cat([x, x1], dim=1))
        y2 = self.upsample3(y2)
        y3 = self.upsample4(y3)
        return torch.cat([y1, y2, y3], dim=1), x3

    def forward_mem(self, y):
        batch, channels, height, width = y.shape
        memory = self.mem.repeat(batch, 1, 1)
        memory_key = memory.transpose(1, 2)
        y_flat = y.view(batch, channels, -1)
        logits = torch.bmm(memory_key, y_flat) / (channels ** 0.5)
        updated = torch.bmm(memory_key.transpose(1, 2), F.softmax(logits, dim=1))
        return updated.view(batch, channels, height, width), logits

    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c < self.cls_thrs] = 0
        c_new[c >= self.cls_thrs] = 1
        return F.interpolate(c_new, scale_factor=4, mode="nearest")

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] % 16 or x.shape[3] % 16:
            raise ValueError("MPCount expects NCHW RGB with height and width divisible by 16")
        y_cat, x3 = self.forward_fe(x)
        y_den = self.den_dec(y_cat)
        y_den, _ = self.forward_mem(y_den)
        c = self.cls_head(x3)
        density = self.den_head(y_den) * self.transform_cls_map_pred(c)
        return self.upsample_d(density), c


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path, digest):
    """Load a raw official state_dict with hash and strict key/shape checks.

    Returns ``(model.eval(), metadata)``. The density map is unscaled; the
    upstream inference convention sums it and divides by 1000.
    """
    checkpoint = Path(path).resolve(strict=True)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(digest)):
        raise ValueError("Expected checkpoint SHA-256 must be 64 hexadecimal characters")
    actual_digest = _sha256(checkpoint)
    if actual_digest.lower() != digest.lower():
        raise ValueError("MPCount checkpoint SHA-256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()):
        raise ValueError("Expected an official raw tensor state_dict")
    model = DGModel_final()
    expected = model.state_dict()
    if len(state) != 147 or len(expected) != 147 or state.keys() != expected.keys():
        raise ValueError("MPCount checkpoint must match the official 147-key state_dict")
    for key, tensor in state.items():
        reference = expected[key]
        if (tensor.layout != torch.strided or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype):
            raise ValueError(f"Invalid checkpoint tensor layout/shape/dtype: {key}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {
        "architecture": "DGModel_final",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": actual_digest,
        "state_tensor_count": len(state),
        "state_keys_verified": True,
        "state_shapes_dtypes_finiteness_verified": True,
        "density_output": "full-resolution signed map; upstream count=sum(map)/1000",
        "pac_gate": "sigmoid probability >= 0.5, nearest-neighbor x4, gates quarter-resolution density before fixed x4 upsample",
        "pretrained_vgg_download": False,
    }
    return model, metadata
