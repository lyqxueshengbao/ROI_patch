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
      - low-frequency branch uses X0 (or [x_mag, x_sin, x_cos] for 4-ch input)
      - high-frequency branch uses Xhf
    Fused by a learned gate g in (0,1): z = (1-g)*z_low + g*z_high

    For 2-channel input: low=ch0, high=ch1
    For 4-channel input: low=ch[0,2,3] (x_mag, x_sin, x_cos), high=ch1 (x_hf)
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        if in_channels == 2:
            low_ch = 1
            high_ch = 1
        elif in_channels == 4:
            # 4-ch: [x_mag, x_hf, x_sin, x_cos]
            # low branch: x_mag, x_sin, x_cos (ch 0, 2, 3)
            # high branch: x_hf (ch 1)
            low_ch = 3
            high_ch = 1
        else:
            raise ValueError(f"HFGatedFusionNet only supports in_channels=2 or 4, got {in_channels}")

        self.low = _Branch(in_channels=low_ch, base_channels=base_channels)
        self.high = _Branch(in_channels=high_ch, base_channels=base_channels)
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
        if self.in_channels == 2:
            x0 = x[:, 0:1]
            xhf = x[:, 1:2]
        else:
            # 4-ch: [x_mag, x_hf, x_sin, x_cos]
            x0 = torch.cat([x[:, 0:1], x[:, 2:4]], dim=1)  # [B, 3, H, W]
            xhf = x[:, 1:2]  # [B, 1, H, W]
        z_low = self.low(x0)
        z_high = self.high(xhf)
        g = self.gate(torch.cat([z_low, z_high], dim=1))
        z = (1.0 - g) * z_low + g * z_high
        return self.head(z)

