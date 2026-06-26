"""Dense BCHW <-> MinkowskiEngine SparseTensor helpers for 2D codecs."""

from __future__ import annotations

import torch

_ME = None


def require_minkowski_engine():
    global _ME
    if _ME is None:
        try:
            import MinkowskiEngine as ME  # noqa: N812
        except ImportError as exc:
            raise ImportError(
                "MinkowskiEngine is required for ME codec variants. "
                "Install ME (CUDA) before using HPCM_*_ME models."
            ) from exc
        _ME = ME
    return _ME


def bchw_to_sparse(x: torch.Tensor, tensor_stride: int = 1):
    """Convert dense [B,C,H,W] to a full-grid SparseTensor."""
    ME = require_minkowski_engine()
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")

    B, C, H, W = x.shape
    device = x.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    spatial = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1).int()
    coords = []
    for b in range(B):
        batch_col = torch.full((H * W, 1), b, device=device, dtype=torch.int32)
        coords.append(torch.cat([batch_col, spatial], dim=1))
    coordinates = torch.cat(coords, dim=0)
    features = x.permute(0, 2, 3, 1).reshape(B * H * W, C).contiguous()
    return ME.SparseTensor(
        features=features,
        coordinates=coordinates,
        tensor_stride=tensor_stride,
        device=device,
    )


def sparse_to_bchw(st, shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Scatter SparseTensor features back to dense [B,C,H,W]."""
    B, C, H, W = shape
    dense = torch.zeros(B, C, H, W, device=st.device, dtype=st.F.dtype)
    coords = st.C.long()
    dense[coords[:, 0], :, coords[:, 1], coords[:, 2]] = st.F
    return dense


def out_hw(h: int, w: int, kernel_size: int, stride: int) -> tuple[int, int]:
    if stride == 1:
        return h, w
    return h // stride, w // stride


def out_hw_transpose(h: int, w: int, stride: int) -> tuple[int, int]:
    return h * stride, w * stride
