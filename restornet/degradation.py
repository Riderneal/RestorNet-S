"""
Physics-matched degradation engine.

Simulates the real inspection-image degradation pipeline described in the
problem statement:
    1. Multiplicative Speckle noise (grainy, can push pixels out of the
       original [0, 1] intensity range).
    2. Additive Gaussian sensor noise.
    3. Bicubic downsampling (spatial resolution reduction).

Used both for on-the-fly training-pair generation and for building the
sample degraded/restored/ground-truth comparisons.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def add_speckle_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Multiplicative speckle noise: I' = I + I * N(0, sigma^2).

    Deliberately NOT clamped before the additive-noise stage, so pixels can
    be pushed outside [0, 1] -- exactly the "out-of-range pixel" behaviour
    called out in the problem statement. Clamping happens once, at the end
    of the full degradation pipeline.
    """
    noise = torch.randn_like(img) * sigma
    return img + img * noise


def add_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return img + torch.randn_like(img) * sigma


def bicubic_downsample(img: torch.Tensor, scale: int) -> torch.Tensor:
    """img: (C, H, W) or (N, C, H, W) tensor in [0, 1]."""
    squeeze = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    h, w = img.shape[-2:]
    out = F.interpolate(
        img, size=(h // scale, w // scale), mode="bicubic", align_corners=False
    )
    return out.squeeze(0) if squeeze else out


def degrade(
    img: torch.Tensor,
    scale: int = 2,
    speckle_sigma: float = 0.15,
    gaussian_sigma: float = 0.02,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Full degradation pipeline matching the problem statement:
    speckle -> additive Gaussian -> bicubic downsample -> clamp.

    Args:
        img: ground-truth image tensor, shape (1, H, W) or (N, 1, H, W),
             values in [0, 1].
        scale: downsampling factor (spatial resolution reduction).
        speckle_sigma: std-dev of the multiplicative speckle term.
        gaussian_sigma: std-dev of the additive Gaussian term.

    Returns:
        Degraded, low-resolution tensor, same rank as input, values
        clamped to [0, 1].
    """
    if rng is not None:
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    noisy = add_speckle_noise(img, speckle_sigma)
    noisy = add_gaussian_noise(noisy, gaussian_sigma)
    low_res = bicubic_downsample(noisy, scale)
    return torch.clamp(low_res, 0.0, 1.0)
