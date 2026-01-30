from __future__ import annotations

import torch
from torch import nn


class _Branch(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z.flatten(1)


class HFGatedFusionNet(nn.Module):
    """
    Dual-branch network:
      - low-frequency branch uses X0
      - high-frequency branch uses Xhf
    Fused by a learned gate g in (0,1): z = (1-g)*z_low + g*z_high
    """

    def __init__(self, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.low = _Branch(in_channels=1, base_channels=base_channels)
        self.high = _Branch(in_channels=1, base_channels=base_channels)
        feat_dim = self.low.out_dim
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0:1]
        xhf = x[:, 1:2]
        z_low = self.low(x0)
        z_high = self.high(xhf)
        g = self.gate(torch.cat([z_low, z_high], dim=1))
        z = (1.0 - g) * z_low + g * z_high
        return self.head(z)

