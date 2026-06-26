"""Dense BCHW wrappers around MinkowskiEngine sparse convolutions (2D codec)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .me_sparse import bchw_to_sparse, out_hw, out_hw_transpose, require_minkowski_engine, sparse_to_bchw


class _MEDenseConvBase(nn.Module):
    """Convert dense -> SparseTensor -> ME op -> dense."""

    out_channels: int

    def me_forward(self, st):
        raise NotImplementedError

    def output_hw(self, h: int, w: int) -> tuple[int, int]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        b, _, h, w = x.shape
        st = bchw_to_sparse(x)
        st = self.me_forward(st)
        oh, ow = self.output_hw(h, w)
        return sparse_to_bchw(st, (b, self.out_channels, oh, ow))


def me_conv1x1(in_ch: int, out_ch: int) -> _MEDenseConvBase:
    ME = require_minkowski_engine()

    class _Op(_MEDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = ME.MinkowskiConvolution(
                in_ch, out_ch, kernel_size=1, stride=1, dimension=2
            )

        def me_forward(self, st):
            return self.conv(st)

        def output_hw(self, h, w):
            return h, w

    return _Op()


def me_conv3x3(in_ch: int, out_ch: int) -> _MEDenseConvBase:
    ME = require_minkowski_engine()

    class _Op(_MEDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = ME.MinkowskiConvolution(
                in_ch, out_ch, kernel_size=3, stride=1, dimension=2
            )

        def me_forward(self, st):
            return self.conv(st)

        def output_hw(self, h, w):
            return h, w

    return _Op()


def me_conv2x2_down(in_ch: int, out_ch: int) -> _MEDenseConvBase:
    ME = require_minkowski_engine()

    class _Op(_MEDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = ME.MinkowskiConvolution(
                in_ch, out_ch, kernel_size=2, stride=2, dimension=2
            )

        def me_forward(self, st):
            return self.conv(st)

        def output_hw(self, h, w):
            return out_hw(h, w, kernel_size=2, stride=2)

    return _Op()


def me_conv4x4_down(in_ch: int, out_ch: int) -> _MEDenseConvBase:
    ME = require_minkowski_engine()

    class _Op(_MEDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = ME.MinkowskiConvolution(
                in_ch, out_ch, kernel_size=4, stride=2, dimension=2
            )

        def me_forward(self, st):
            return self.conv(st)

        def output_hw(self, h, w):
            return out_hw(h, w, kernel_size=4, stride=2)

    return _Op()


def me_deconv2x2_up(in_ch: int, out_ch: int) -> _MEDenseConvBase:
    ME = require_minkowski_engine()

    class _Op(_MEDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = ME.MinkowskiConvolutionTranspose(
                in_ch, out_ch, kernel_size=2, stride=2, dimension=2
            )

        def me_forward(self, st):
            return self.conv(st)

        def output_hw(self, h, w):
            return out_hw_transpose(h, w, stride=2)

    return _Op()


def me_deconv4x4_up(in_ch: int, out_ch: int) -> _MEDenseConvBase:
    ME = require_minkowski_engine()

    class _Op(_MEDenseConvBase):
        def __init__(self):
            super().__init__()
            self.out_channels = out_ch
            self.conv = ME.MinkowskiConvolutionTranspose(
                in_ch, out_ch, kernel_size=4, stride=2, dimension=2
            )

        def me_forward(self, st):
            return self.conv(st)

        def output_hw(self, h, w):
            return out_hw_transpose(h, w, stride=2)

    return _Op()


class me_pconv3x3(nn.Module):
    """Partial 3x3 conv on the first N1 channels (ME backend)."""

    def __init__(self, n: int, n1: int):
        super().__init__()
        self.n = n
        self.n1 = n1
        self.pconv = me_conv3x3(n1, n1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.n1, self.n - self.n1], dim=1)
        x1 = self.pconv(x1)
        return torch.cat((x1, x2), dim=1)
