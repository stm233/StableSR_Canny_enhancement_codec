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


def psnr_continuous(
    pred: torch.Tensor,
    target: torch.Tensor,
    peak: float = 255.0,
) -> torch.Tensor:
    """
    pred, target in [0, 1].
    PSNR = 10 * log10(peak^2 / MSE_peak),  MSE_peak = mean((pred-target)^2) * peak^2.
    """
    mse_01 = (pred - target).pow(2).mean()
    mse_peak = mse_01 * (peak ** 2)
    return 10 * torch.log10((peak ** 2) / mse_peak.clamp(min=1e-10))


def compute_dt_canny_psnr(
    x_hat: torch.Tensor,
    gt_r: torch.Tensor,
    gt_canny: torch.Tensor,
    edge_threshold: float = 0.5,
) -> tuple[float, float]:
    """Return (PSNR on inverted R map, PSNR on binarized edge Canny)."""
    from src.utils.distance_transform import inverted_r_to_edge_uint8

    psnr_dt = psnr_continuous(x_hat, gt_r, peak=255.0).item()

    gt_edge = (gt_canny >= 0.5).float()
    recon_edge = torch.from_numpy(
        inverted_r_to_edge_uint8(x_hat, threshold=edge_threshold).astype("float32") / 255.0
    ).view_as(gt_edge).to(gt_edge.device)
    psnr_canny = psnr_continuous(recon_edge, gt_edge, peak=255.0).item()
    return psnr_dt, psnr_canny
