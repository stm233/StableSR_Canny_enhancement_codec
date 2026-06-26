"""Residual blocks with spconv convolutions (dense in/out)."""

import torch.nn as nn

from .spconv_conv import spconv_conv1x1, spconv_pconv3x3


class SpconvPConvRB(nn.Module):
    def __init__(self, n: int = 192, partial_ratio: int = 4, mlp_ratio: int = 2, act=nn.LeakyReLU):
        super().__init__()
        n1 = n // partial_ratio
        middle_ch = n * mlp_ratio
        self.branch = nn.Sequential(
            spconv_pconv3x3(n, n1),
            spconv_conv1x1(n, middle_ch),
            act(inplace=True),
            spconv_conv1x1(middle_ch, n),
        )

    def forward(self, x):
        return x + self.branch(x)
