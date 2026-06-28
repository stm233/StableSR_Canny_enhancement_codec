"""I-frame Canny1ch codec with canny64_lossy conditional encoder (H/2 + H/4 fusion).

g_a / g_s: partial spconv (encoder first two stages, decoder last two stages).
h_a / h_s / cond_g_a: dense conv on hyper path; cond_g_a shares g_a partial-spconv stem.
"""

from __future__ import annotations

import torch
from torch import nn

from src.layers import conv1x1

from .base import BB as basemodel
from .HPCM_Base import CrossAttentionCell, y_spatial_prior_s1_s2
from .HPCM_Canny1ch import LATENT_M, LATENT_N, h_a_128, h_s_128
from .HPCM_Canny1ch_Spconv import (
    g_a_1ch_partial_spconv,
    g_s_1ch_partial_spconv,
)
from .codec_fusion import (
    extract_ga_encoder_cond_feats,
    g_a_forward,
    g_s_decode_multiscale,
)

__all__ = ["HPCM", "LATENT_M", "LATENT_N"]

C32, C64, C128 = 32, 64, 128


class HPCM(basemodel):
    """Compress target Canny (e.g. 256x256) conditioned on upsampled canny64_lossy."""

    def __init__(self, m: int = LATENT_M, n: int = LATENT_N):
        super().__init__(n)
        self.M = m
        self.N = n

        self.g_a = g_a_1ch_partial_spconv(m)
        self.g_s = g_s_1ch_partial_spconv(m)
        self.h_a = h_a_128(m, n)
        self.h_s = h_s_128(m, n)
        self.cond_g_a = g_a_1ch_partial_spconv(m)

        self.cond_fuse = nn.ModuleDict({
            "enc_cond_h2": conv1x1(C32, C32),
            "enc_cond_h4": conv1x1(C64, C64),
            "dec_cond_h4_f2": conv1x1(C64, C128),
            "dec_cond_h4_f1": conv1x1(C64, C64),
            "dec_cond_h2": conv1x1(C32, C32),
        })
        self._init_cond_fuse()

        ctx_ch = m * 2
        self.y_spatial_prior_adaptor_list = nn.ModuleList(
            conv1x1(3 * m, 3 * m) for _ in range(3)
        )
        self.y_spatial_prior = y_spatial_prior_s1_s2(m)
        self.adaptive_params_list = nn.ParameterList([
            nn.Parameter(torch.ones((1, m * 3, 1, 1)), requires_grad=True)
            for _ in range(3)
        ])
        self.attn = CrossAttentionCell(ctx_ch, ctx_ch, window_size=8, kernel_size=1)
        self.context_net = conv1x1(ctx_ch, ctx_ch)

    def _init_cond_fuse(self) -> None:
        with torch.no_grad():
            for m in self.cond_fuse.values():
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()

    def _unpack(self, x, cond=None):
        if isinstance(x, dict):
            cond = x.get("cond", cond)
            x = x["input"]
        return x, cond

    def _cond_feats(self, cond: torch.Tensor | None) -> dict[str, torch.Tensor] | None:
        if cond is None:
            return None
        return extract_ga_encoder_cond_feats(self.cond_g_a, cond)

    def forward(self, x, cond=None, training=None):
        x, cond = self._unpack(x, cond)
        if training is None:
            training = self.training

        cond_feats = self._cond_feats(cond)
        y = g_a_forward(self.g_a, x, None, None, cond_feats, self.cond_fuse)
        z = self.h_a(y)

        if training:
            z_res = z - self.means_hyper
            z_hat = self.ste_round(z_res) + self.means_hyper
            z_likelihoods = self.entropy_estimation(self.add_noise(z_res), self.scales_hyper)
        else:
            z_res_hat = torch.round(z - self.means_hyper)
            z_hat = z_res_hat + self.means_hyper
            z_likelihoods = self.entropy_estimation(z_res_hat, self.scales_hyper)

        params = self.h_s(z_hat)
        y_res, y_q, y_hat, scales_y = self.forward_hpcm(y, params)
        _, x_hat = g_s_decode_multiscale(
            self.g_s, y_hat, None, None, cond_feats, self.cond_fuse
        )
        x_hat = x_hat.clamp(0.0, 1.0)

        if training:
            y_likelihoods = self.entropy_estimation(self.add_noise(y_res), scales_y)
        else:
            y_res_hat = torch.round(y_res)
            y_likelihoods = self.entropy_estimation(y_res_hat, scales_y)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def forward_hpcm(self, y, common_params, write=False):
        b, c, h, w = y.size()
        dtype = common_params.dtype
        device = common_params.device

        scales_all, means_all = common_params.chunk(2, 1)
        context_next = common_params
        mask_list = self.get_mask_four_parts_two_groups(b, c, h, w, dtype, device)

        y_res_list, y_q_list, y_hat_list, scale_list = [], [], [], []

        for i in range(4):
            if i == 0:
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(
                    y, scales_all, means_all, mask_list[i]
                )
                y_res_list.append(y_res_0)
                y_q_list.append(y_q_0)
                y_hat_list.append(y_hat_0)
                scale_list.append(s_hat_0)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list), dim=0)
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = self.y_spatial_prior(
                    self.y_spatial_prior_adaptor_list[i - 1](params),
                    self.adaptive_params_list[i - 1],
                )
                context_next = self.attn(context, context_next)
                scales, means = context.chunk(2, 1)
                y_res_i, y_q_i, y_hat_i, s_hat_i = self.process_with_mask(
                    y, scales, means, mask_list[i]
                )
                y_res_list.append(y_res_i)
                y_q_list.append(y_q_i)
                y_hat_list.append(y_hat_i)
                scale_list.append(s_hat_i)

        y_res = torch.sum(torch.stack(y_res_list), dim=0)
        y_q = torch.sum(torch.stack(y_q_list), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list), dim=0)

        if write:
            return y_q_list, scale_list
        return y_res, y_q, y_hat, scales_hat

    def compress(self, x, cond=None):
        from src.entropy_models import ubransEncoder

        x, cond = self._unpack(x, cond)
        cond_feats = self._cond_feats(cond)
        y = g_a_forward(self.g_a, x, None, None, cond_feats, self.cond_fuse)
        z = self.h_a(y)
        z_res_hat = torch.round(z - self.means_hyper)
        indexes_z = self.build_indexes_z(z_res_hat.size())

        encoder_z = ubransEncoder()
        self.compress_symbols(
            z_res_hat, indexes_z,
            self.quantized_cdf_z.cpu().numpy(),
            self.cdf_length_z.cpu().numpy(),
            self.offset_z.cpu().numpy(),
            encoder_z,
        )
        z_string = encoder_z.flush()
        z_hat = z_res_hat + self.means_hyper

        params = self.h_s(z_hat)
        y_q_list, scale_list = self.forward_hpcm(y, params, write=True)

        encoder_y = ubransEncoder()
        self.compress_y_two_group_lite(y_q_list, scale_list, encoder_y)
        y_string = encoder_y.flush()
        return {"strings": [y_string, z_string], "shape": z_res_hat.size()[2:]}

    def decompress(self, strings, shape, cond=None):
        from src.entropy_models import ubransDecoder

        cond_feats = self._cond_feats(cond)
        device = self.quantized_cdf_z.device
        output_size = (1, self.scales_hyper.size(1), *shape)
        indexes_z = self.build_indexes_z(output_size).to(device)

        decoder_z = ubransDecoder()
        decoder_z.set_stream(strings[1])
        z_res_hat = self.decompress_symbols(
            indexes_z,
            self.quantized_cdf_z.cpu().numpy(),
            self.cdf_length_z.cpu().numpy(),
            self.offset_z.cpu().numpy(),
            decoder_z,
        )
        z_hat = z_res_hat + self.means_hyper

        params = self.h_s(z_hat)
        decoder_y = ubransDecoder()
        decoder_y.set_stream(strings[0])
        y_hat = self.decompress_hpcm(params, decoder_y)
        _, x_hat = g_s_decode_multiscale(
            self.g_s, y_hat, None, None, cond_feats, self.cond_fuse
        )
        return {"x_hat": x_hat.clamp_(0, 1)}

    def decompress_with_latent(self, strings, shape, cond=None):
        from src.entropy_models import ubransDecoder

        cond_feats = self._cond_feats(cond)
        device = self.quantized_cdf_z.device
        output_size = (1, self.scales_hyper.size(1), *shape)
        indexes_z = self.build_indexes_z(output_size).to(device)

        decoder_z = ubransDecoder()
        decoder_z.set_stream(strings[1])
        z_res_hat = self.decompress_symbols(
            indexes_z,
            self.quantized_cdf_z.cpu().numpy(),
            self.cdf_length_z.cpu().numpy(),
            self.offset_z.cpu().numpy(),
            decoder_z,
        )
        z_hat = z_res_hat + self.means_hyper

        params = self.h_s(z_hat)
        decoder_y = ubransDecoder()
        decoder_y.set_stream(strings[0])
        y_hat = self.decompress_hpcm(params, decoder_y)
        _, x_hat = g_s_decode_multiscale(
            self.g_s, y_hat, None, None, cond_feats, self.cond_fuse
        )
        return {"x_hat": x_hat.clamp_(0, 1), "y_hat": y_hat}

    def decompress_hpcm(self, common_params, decoder_y):
        scales, means = common_params.chunk(2, 1)
        dtype = means.dtype
        device = means.device
        b, c, h, w = means.size()

        context_next = common_params
        mask_list = self.get_mask_four_parts_two_groups(b, c, h, w, dtype, device)
        y_hat_so_far = None

        for i in range(4):
            if i == 0:
                y_hat_curr_step = self.decompress_y_two_group_lite_step(
                    scales, means, mask_list[i], decoder_y
                )
                y_hat_so_far = y_hat_curr_step
            else:
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = self.y_spatial_prior(
                    self.y_spatial_prior_adaptor_list[i - 1](params),
                    self.adaptive_params_list[i - 1],
                )
                context_next = self.attn(context, context_next)
                scales, means = context.chunk(2, 1)
                y_hat_curr_step = self.decompress_y_two_group_lite_step(
                    scales, means, mask_list[i], decoder_y
                )
                y_hat_so_far = y_hat_so_far + y_hat_curr_step

        return y_hat_so_far
