"""L1 distance transform features for edge maps and simple decode post-process."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_bf


def edge_binary_from_tensor(edge: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """[H,W] or [1,H,W] float in [0,1] -> bool edge mask."""
    if edge.dim() == 3:
        edge = edge.squeeze(0)
    return (edge.detach().cpu().numpy() > threshold)


def canny_to_dt_rgb(
    edge01: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Binary Canny -> 3ch tensor [3, H, W] in [0, 1]:
      R = L1 distance to nearest edge (normalized by H+W)
      G = nearest edge column / (W-1)
      B = nearest edge row / (H-1)
    """
    if isinstance(edge01, torch.Tensor):
        edge01 = edge01.detach().cpu().numpy()
    if edge01.ndim == 3:
        edge01 = edge01.squeeze(0)

    h, w = edge01.shape
    edge = edge01 > threshold
    norm_d = float(h + w)
    norm_x = max(w - 1, 1)
    norm_y = max(h - 1, 1)

    if not edge.any():
        r = np.ones((h, w), dtype=np.float32)
        g = np.zeros((h, w), dtype=np.float32)
        b = np.zeros((h, w), dtype=np.float32)
        return torch.from_numpy(np.stack([r, g, b], axis=0))

    inv = (~edge).astype(np.uint8)
    dist, indices = distance_transform_bf(
        inv, metric="cityblock", return_distances=True, return_indices=True
    )
    row_idx = indices[0].astype(np.float32)
    col_idx = indices[1].astype(np.float32)

    r = dist.astype(np.float32) / norm_d
    g = col_idx / norm_x
    b = row_idx / norm_y
    return torch.from_numpy(np.stack([r, g, b], axis=0))


def distance_to_edge_uint8(
    dist: torch.Tensor | np.ndarray,
    h: int | None = None,
    w: int | None = None,
    denorm: bool = True,
    tol: float = 0.5,
) -> np.ndarray:
    """
  Decode post-process: distance 0 -> edge white (255), else black (0).

  dist: [1,H,W] or [H,W], normalized distance (R channel) if denorm uses H+W scale.
    """
    if isinstance(dist, torch.Tensor):
        dist = dist.detach().cpu().numpy()
    if dist.ndim == 3:
        dist = dist.squeeze(0)

    if denorm:
        if h is None or w is None:
            h, w = dist.shape
        dist_px = dist * float(h + w)
    else:
        dist_px = dist

    edge = (dist_px <= tol).astype(np.uint8) * 255
    return edge


def dt_rgb_to_distance_uint8(rgb: torch.Tensor, h: int, w: int, tol: float = 0.5) -> np.ndarray:
    """GT distance channel -> binary edge PNG (for metric vs original Canny)."""
    return distance_to_edge_uint8(rgb[0], h=h, w=w, denorm=True, tol=tol)
