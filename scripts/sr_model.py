"""SR inference architecture adapted from wbshu/SR (MIT, copyright 2026 wbshu).

Source: Model/vgg_19.py at 6c5f66c823dd86dcb1329f1b007ba369e622036b.
Only architecture: no implicit downloads or checkpoint deserialization.
The upstream upsample_bilinear uses align_corners=True, preserved here.
See third_party/sr-LICENSE for the upstream license.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SRDensity(nn.Module):
    def __init__(self):
        super().__init__()
        configuration = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256,
                         "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
        layers, channels = [], 3
        for value in configuration:
            if value == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([nn.Conv2d(channels, value, 3, padding=1),
                               nn.BatchNorm2d(value), nn.ReLU(inplace=True)])
                channels = value
        self.features = nn.Sequential(*layers)
        self.reg_layer = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 1, 1))

    def forward(self, inputs):
        features = self.features(inputs)
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=True)
        return self.reg_layer(features).abs()
