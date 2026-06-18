"""L1 distance transform features for edge maps and simple decode post-process."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def _l1_dt_opencv(inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L1 DT on uint8 inv (edge=0, non-edge=1). Returns dist, row_idx, col_idx."""
    dist, labels = cv2.distanceTransformWithLabels(
        inv,
        cv2.DIST_L1,
        cv2.DIST_MASK_PRECISE,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    h, w = inv.shape
    row_idx = np.zeros((h, w), dtype=np.float32)
    col_idx = np.zeros((h, w), dtype=np.float32)

    seed_y, seed_x = np.where(inv == 0)
    if seed_y.size == 0:
        return dist.astype(np.float32), row_idx, col_idx

    valid = labels > 0
    lid = labels[valid] - 1
    row_idx[valid] = seed_y[lid]
    col_idx[valid] = seed_x[lid]

    edge_mask = inv == 0
    row_idx[edge_mask] = seed_y.astype(np.float32)
    col_idx[edge_mask] = seed_x.astype(np.float32)
    return dist.astype(np.float32), row_idx, col_idx


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
      R = 1.0 on edge; else L1 distance / (H+W)
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
        r = np.zeros((h, w), dtype=np.float32)
        g = np.zeros((h, w), dtype=np.float32)
        b = np.zeros((h, w), dtype=np.float32)
        return torch.from_numpy(np.stack([r, g, b], axis=0))

    inv = (~edge).astype(np.uint8)
    dist, row_idx, col_idx = _l1_dt_opencv(inv)

    r = dist.astype(np.float32) / norm_d
    r[edge] = 1.0
    g = col_idx / norm_x
    b = row_idx / norm_y
    return torch.from_numpy(np.stack([r, g, b], axis=0))


def _as_hw_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """[1,1,H,W] / [1,H,W] / [H,W] -> [H,W] float numpy."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def inverted_r_to_edge_uint8(
    r_hat: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Inverted R decode: edge=1 on edge pixels, else distance -> edge iff R_hat >= threshold."""
    return continuous_canny_to_edge_uint8(r_hat, threshold=threshold)


def distance_to_edge_uint8(
    dist: torch.Tensor | np.ndarray,
    h: int | None = None,
    w: int | None = None,
    denorm: bool = True,
    tol: float = 0.5,
) -> np.ndarray:
    """
    Legacy DT decode (edge=0 encoding): dist_px = R_hat * (H+W); edge iff dist_px <= tol.

    For inverted R (edge=1), use inverted_r_to_edge_uint8 instead.
    """
    if isinstance(dist, torch.Tensor):
        dist = dist.detach().cpu().numpy()
    dist = _as_hw_numpy(dist)

    if denorm:
        if h is None or w is None:
            h, w = dist.shape
        dist_px = dist * float(h + w)
    else:
        dist_px = dist

    edge = (dist_px <= tol).astype(np.uint8) * 255
    return edge


def dt_r_hat_to_canny_uint8(
    r_hat: torch.Tensor,
    h: int,
    w: int,
    threshold: float = 0.5,
    tol_px: float = 0.5,
) -> np.ndarray:
    """HPCM_DT1ch: decoder output [1,H,W] inverted R -> binary Canny PNG."""
    del h, w, tol_px  # inverted R uses threshold only
    return inverted_r_to_edge_uint8(r_hat, threshold=threshold)


def dt_rgb_to_distance_uint8(rgb: torch.Tensor, h: int, w: int, threshold: float = 0.5) -> np.ndarray:
    """GT inverted R channel -> binary edge PNG."""
    del h, w
    return inverted_r_to_edge_uint8(rgb[0], threshold=threshold)


def continuous_canny_to_edge_uint8(
    x: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Lossy codec output [0,1] -> binary Canny (edge=255, bg=0)."""
    x = _as_hw_numpy(x)
    return (x >= threshold).astype(np.uint8) * 255
