"""Tensor helpers shared across codecs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def get_scale_table(min_val: float = 0.12, max_val: float = 64.0, levels: int = 60) -> torch.Tensor:
    return torch.Tensor(np.exp(np.linspace(np.log(min_val), np.log(max_val), levels)))


def pad(x: torch.Tensor, p: int = 2**6) -> torch.Tensor:
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )
