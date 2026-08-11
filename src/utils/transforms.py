"""Colour transforms used by the official DCVC-RT evaluation pipeline.

All neural codec tensors use full-range BT.709 YCbCr 4:4:4 in ``[0, 1]``.
Task networks continue to receive RGB tensors in ``[0, 1]``.
"""

from __future__ import annotations

import torch


BT709_WEIGHTS = (0.2126, 0.7152, 0.0722)


def rgb2ycbcr(rgb: torch.Tensor, is_bgr: bool = False) -> torch.Tensor:
    """Convert full-range RGB/BGR to full-range BT.709 YCbCr 4:4:4."""
    if rgb.shape[-3] != 3:
        raise ValueError(f"Expected three colour channels, got shape {tuple(rgb.shape)}")
    if is_bgr:
        blue, green, red = rgb.chunk(3, dim=-3)
    else:
        red, green, blue = rgb.chunk(3, dim=-3)
    kr, kg, kb = BT709_WEIGHTS
    luminance = kr * red + kg * green + kb * blue
    cb = 0.5 * (blue - luminance) / (1.0 - kb) + 0.5
    cr = 0.5 * (red - luminance) / (1.0 - kr) + 0.5
    return torch.cat((luminance, cb, cr), dim=-3).clamp(0.0, 1.0)


def ycbcr2rgb(
    ycbcr: torch.Tensor,
    is_bgr: bool = False,
    clamp: bool = True,
) -> torch.Tensor:
    """Convert full-range BT.709 YCbCr 4:4:4 to full-range RGB/BGR."""
    if ycbcr.shape[-3] != 3:
        raise ValueError(f"Expected three colour channels, got shape {tuple(ycbcr.shape)}")
    luminance, cb, cr = ycbcr.chunk(3, dim=-3)
    kr, kg, kb = BT709_WEIGHTS
    red = luminance + (2.0 - 2.0 * kr) * (cr - 0.5)
    blue = luminance + (2.0 - 2.0 * kb) * (cb - 0.5)
    green = (luminance - kr * red - kb * blue) / kg
    channels = (blue, green, red) if is_bgr else (red, green, blue)
    rgb = torch.cat(channels, dim=-3)
    return rgb.clamp(0.0, 1.0) if clamp else rgb
