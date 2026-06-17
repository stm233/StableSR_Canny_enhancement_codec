"""HPCM Canny codec: g_a/h_a/h_s same as HPCM_Base (3ch in), g_s outputs 1ch Canny."""

import torch
from torch import nn

from .base import BB as basemodel
from .HPCM_Base import CrossAttentionCell, g_a, h_a, h_s, y_spatial_prior_s1_s2
from src.layers import PConvRB, conv1x1, deconv2x2_up, deconv4x4_up


class g_s_1ch(nn.Module):
    """HPCM_Base.g_s with internal width ÷3 (384→128, 192→64, 96→32), output 1ch."""

    def __init__(self):
        super().__init__()
        mlp_ratio = 4
        partial_ratio = 4
        C128, C64, C32 = 384 // 3, 192 // 3, 96 // 3  # 128, 64, 32
        self.branch = nn.Sequential(
            deconv2x2_up(320, C128),
            PConvRB(C128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(C128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(C128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(C128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            deconv2x2_up(C128, C64),
            PConvRB(C64, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(C64, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            deconv2x2_up(C64, C32),
            PConvRB(C32, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(C32, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            deconv4x4_up(C32, 1),
        )

    def forward(self, x):
        return self.branch(x)


class HPCM(basemodel):
    """3ch Canny in (R=G=B), 1ch Canny out; Lite entropy (4 checkerboard steps)."""

    def __init__(self, M=320, N=256):
        super().__init__(N)

        self.g_a = g_a()
        self.g_s = g_s_1ch()
        self.h_a = h_a()
        self.h_s = h_s()

        self.y_spatial_prior_adaptor_list = nn.ModuleList(
            conv1x1(3 * M, 3 * M) for _ in range(3)
        )
        self.y_spatial_prior = y_spatial_prior_s1_s2(M)
        self.adaptive_params_list = nn.ParameterList([
            nn.Parameter(torch.ones((1, M * 3, 1, 1)), requires_grad=True)
            for _ in range(3)
        ])
        self.attn = CrossAttentionCell(320 * 2, 320 * 2, window_size=8, kernel_size=1)
        self.context_net = conv1x1(2 * M, 2 * M)

    def forward(self, x, training=None):
        if training is None:
            training = self.training

        y = self.g_a(x)
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
        x_hat = self.g_s(y_hat).clamp(0.0, 1.0)

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
        B, C, H, W = y.size()
        dtype = common_params.dtype
        device = common_params.device

        scales_all, means_all = common_params.chunk(2, 1)
        context_next = common_params
        mask_list = self.get_mask_four_parts_two_groups(B, C, H, W, dtype, device)

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

    def compress(self, x):
        from src.entropy_models import ubransEncoder

        y = self.g_a(x)
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

    def decompress(self, strings, shape):
        from src.entropy_models import ubransDecoder

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
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def decompress_hpcm(self, common_params, decoder_y):
        scales, means = common_params.chunk(2, 1)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()

        context_next = common_params
        mask_list = self.get_mask_four_parts_two_groups(B, C, H, W, dtype, device)
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
