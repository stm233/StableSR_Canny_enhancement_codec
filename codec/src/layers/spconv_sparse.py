"""Dense BCHW <-> spconv SparseConvTensor helpers for 2D codecs."""

from __future__ import annotations

import torch

_SPCONV = None


def require_spconv():
    global _SPCONV
    if _SPCONV is None:
        try:
            import spconv.pytorch as spconv  # noqa: N812
        except ImportError as exc:
            raise ImportError(
                "spconv is required for Spconv codec variants. "
                "Install a wheel matching your CUDA, e.g. "
                "pip install spconv-cu118 (see codec/scripts/hpcm/install_spconv.sh)."
            ) from exc
        _SPCONV = spconv
    return _SPCONV


def bchw_to_sparse(x: torch.Tensor):
    """Convert dense [B,C,H,W] to a full-grid SparseConvTensor."""
    spconv = require_spconv()
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")

    b, c, h, w = x.shape
    device = x.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.int32),
        torch.arange(w, device=device, dtype=torch.int32),
        indexing="ij",
    )
    spatial = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)
    coords = []
    for bi in range(b):
        batch_col = torch.full((h * w, 1), bi, device=device, dtype=torch.int32)
        coords.append(torch.cat([batch_col, spatial], dim=1))
    indices = torch.cat(coords, dim=0).contiguous()
    features = x.permute(0, 2, 3, 1).reshape(b * h * w, c).contiguous()
    return spconv.SparseConvTensor(features, indices, [h, w], b)


def sparse_to_bchw(st, shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Scatter SparseConvTensor features back to dense [B,C,H,W]."""
    b, c, h, w = shape
    dense = torch.zeros(b, c, h, w, device=st.features.device, dtype=st.features.dtype)
    idx = st.indices.long()
    dense[idx[:, 0], :, idx[:, 1], idx[:, 2]] = st.features
    return dense


def out_hw(h: int, w: int, kernel_size: int, stride: int) -> tuple[int, int]:
    if stride == 1:
        return h, w
    return h // stride, w // stride


def out_hw_transpose(h: int, w: int, stride: int) -> tuple[int, int]:
    return h * stride, w * stride
