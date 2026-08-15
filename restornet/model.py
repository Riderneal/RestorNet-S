"""
RestorNet-S -- Structure-Preserving Joint Denoising & Super-Resolution.

Architecture summary (see README.md for the full design rationale):
    - Shallow feature extraction (single conv).
    - A stack of Residual Dense Blocks (RDB) with Channel Attention (CA),
      inspired by RCAN / SwinIR-style residual-in-residual design.
    - A lightweight "periodicity" branch: a strided/dilated conv path whose
      output is added back into the main trunk to help the network preserve
      repeating device structures (e.g. DRAM columns, FinFET fins) instead
      of smoothing them away.
    - Sub-pixel (PixelShuffle) upsampling head that performs the resolution
      recovery jointly with denoising (single forward pass).

The network operates on single-channel (grayscale) inspection images, as
specified in the KLA track description.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 4), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) with channel attention at its output."""

    def __init__(self, channels: int, growth: int = 32, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = channels
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, growth, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_ch += growth
        self.fuse = nn.Conv2d(in_ch, channels, 1)
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
        fused = self.fuse(torch.cat(feats, dim=1))
        fused = self.ca(fused)
        return x + fused


class PeriodicityBranch(nn.Module):
    """
    Cheap dilated-conv branch that widens the receptive field along
    periodic structures without extra downsampling, so repeating patterns
    (DRAM columns / FinFET fins) get reinforced rather than blurred out.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


class RestorNetS(nn.Module):
    """
    Joint denoising + super-resolution network.

    Args:
        in_channels: input channels (1 = grayscale inspection images).
        base_channels: trunk width.
        n_rdb: number of Residual Dense Blocks in the trunk.
        scale: super-resolution upscaling factor (2 or 4).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        n_rdb: int = 6,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.body = nn.ModuleList(
            [ResidualDenseBlock(base_channels) for _ in range(n_rdb)]
        )
        self.periodicity = PeriodicityBranch(base_channels)
        self.body_fuse = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # Sub-pixel upsampling head (supports x2 or x4 via repeated x2 stages)
        up_layers = []
        n_up = 1 if scale == 2 else 2 if scale == 4 else 0
        for _ in range(max(n_up, 0)):
            up_layers += [
                nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*up_layers) if up_layers else nn.Identity()

        self.tail = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        # Learnable global residual: bicubic-upsampled input is added back
        # so the network only has to learn the *correction*, which speeds
        # up convergence and stabilizes training at low sample counts.
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )

        feat = self.head(x)
        trunk = feat
        for block in self.body:
            trunk = block(trunk)
        trunk = self.periodicity(trunk)
        trunk = self.body_fuse(trunk) + feat  # residual-in-residual

        up = self.upsample(trunk)
        out = self.tail(up)
        return torch.clamp(out + base, 0.0, 1.0)

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(scale: int = 2, base_channels: int = 48, n_rdb: int = 6) -> RestorNetS:
    return RestorNetS(
        in_channels=1, base_channels=base_channels, n_rdb=n_rdb, scale=scale
    )


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(1, 1, 128, 128)
    y = m(x)
    print(f"params: {m.count_parameters():,}")
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
