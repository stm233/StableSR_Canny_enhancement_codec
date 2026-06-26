"""Dense BCHW wrappers around spconv 2D sparse convolutions."""

from __future__ import annotations

import torch
import torch.nn as nn

from .spconv_sparse import bchw_to_sparse, require_spconv, sparse_to_bchw


class _SpconvDenseConvBase(nn.Module):
    """Convert dense -> SparseConvTensor -> spconv op -> dense."""

    out_channels: int

    def sparse_forward(self, st):
        raise NotImplementedError

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        b, _, _, _ = x.shape
        st = bchw_to_sparse(x)
        st = self.sparse_forward(st)
        oh, ow = int(st.spatial_shape[0]), int(st.spatial_shape[1])
        return sparse_to_bchw(st, (b, self.out_channels, oh, ow))


def spconv_conv1x1(in_ch: int, out_ch: int) -> _SpconvDenseConvBase:
    spconv = require_spconv()

    class _Op(_SpconvDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = spconv.SubMConv2d(in_ch, out_ch, kernel_size=1, bias=True)

        def sparse_forward(self, st):
            return self.conv(st)

    return _Op()


def spconv_conv3x3(in_ch: int, out_ch: int) -> _SpconvDenseConvBase:
    spconv = require_spconv()

    class _Op(_SpconvDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = spconv.SubMConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

        def sparse_forward(self, st):
            return self.conv(st)

    return _Op()


def spconv_conv2x2_down(in_ch: int, out_ch: int) -> _SpconvDenseConvBase:
    spconv = require_spconv()

    class _Op(_SpconvDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = spconv.SparseConv2d(
                in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True
            )

        def sparse_forward(self, st):
            return self.conv(st)

    return _Op()


def spconv_conv4x4_down(in_ch: int, out_ch: int) -> _SpconvDenseConvBase:
    spconv = require_spconv()

    class _Op(_SpconvDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = spconv.SparseConv2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True
            )

        def sparse_forward(self, st):
            return self.conv(st)

    return _Op()


def spconv_deconv2x2_up(in_ch: int, out_ch: int) -> _SpconvDenseConvBase:
    spconv = require_spconv()

    class _Op(_SpconvDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = spconv.SparseConvTranspose2d(
                in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True
            )

        def sparse_forward(self, st):
            return self.conv(st)

    return _Op()


def spconv_deconv4x4_up(in_ch: int, out_ch: int) -> _SpconvDenseConvBase:
    spconv = require_spconv()

    class _Op(_SpconvDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = spconv.SparseConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True
            )

        def sparse_forward(self, st):
            return self.conv(st)

    return _Op()


class spconv_pconv3x3(nn.Module):
    """Partial 3x3 conv on the first N1 channels (spconv backend)."""

    def __init__(self, n: int, n1: int):
        super().__init__()
        self.n = n
        self.n1 = n1
        self.pconv = spconv_conv3x3(n1, n1)

    def forward(self, x):
        x1, x2 = x.split([self.n1, self.n - self.n1], dim=1)
        x1 = self.pconv(x1)
        return torch.cat((x1, x2), dim=1)
