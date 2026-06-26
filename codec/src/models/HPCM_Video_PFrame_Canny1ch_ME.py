"""P-frame video codec on HPCM_Canny1ch_ME (MinkowskiEngine encoder/decoder).

Ablation variant: iframe_codec + trainable codec use ME conv in g_a/g_s/h_a/h_s.
Temporal fusion conv1x1 layers remain dense (same as HPCM_Video_PFrame_Canny1ch).
"""

from __future__ import annotations

import torch
from torch import nn

from src.layers import conv1x1

from . import HPCM_Canny1ch_ME
from .codec_fusion import (
    g_a_forward,
    g_s_decode_multiscale,
    lossy_iframe_ref_bundle,
    ref_feats_from_latent,
)

M = HPCM_Canny1ch_ME.LATENT_M
C32, C64, C128 = 32, 64, 128
CTX_CH = M * 2


class HPCM(nn.Module):
    """DCVC-style P-frame for Canny1ch ME I-frame codec (M=N=128)."""

    LATENT_CH = M

    def __init__(self, use_lossy_ref: bool = True):
        super().__init__()
        self.use_lossy_ref = use_lossy_ref
        self.iframe_codec = HPCM_Canny1ch_ME.HPCM()
        self.codec = HPCM_Canny1ch_ME.HPCM()

        self.fuse = nn.ModuleDict({
            "enc_f1": conv1x1(C64, C32),
            "enc_f2": conv1x1(C128, C64),
            "enc_f3": conv1x1(C128, C128),
            "dec_f1": conv1x1(C64, C64),
            "dec_f2": conv1x1(C128, C128),
        })
        self.temporal_ctx_conv = conv1x1(M, CTX_CH)
        self.context_merge = conv1x1(CTX_CH * 2, CTX_CH)
        self._init_fusion()

    def _init_fusion(self) -> None:
        with torch.no_grad():
            for m in self.fuse.values():
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()
            self.temporal_ctx_conv.weight.zero_()
            if self.temporal_ctx_conv.bias is not None:
                self.temporal_ctx_conv.bias.zero_()
            self.context_merge.weight.zero_()
            if self.context_merge.bias is not None:
                self.context_merge.bias.zero_()
            for i in range(CTX_CH):
                self.context_merge.weight[i, i, 0, 0] = 1.0

    def load_iframe_checkpoint(self, path: str, map_location=None) -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.iframe_codec.load_state_dict(ckpt, strict=True)
        for p in self.iframe_codec.parameters():
            p.requires_grad = False
        self.iframe_codec.eval()

    def load_p_codec_checkpoint(self, path: str, map_location=None) -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.codec.load_state_dict(ckpt, strict=True)

    def load_resume_checkpoint(self, path: str, map_location=None) -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt, strict=True)
        for p in self.iframe_codec.parameters():
            p.requires_grad = False
        self.iframe_codec.eval()

    @staticmethod
    def _to_1ch(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x
        return x[:, :1]

    @staticmethod
    def _curr_canny(p_input: torch.Tensor) -> torch.Tensor:
        if p_input.size(1) == 1:
            return p_input
        return p_input[:, 2:3]

    def _prev_ref_feats(self, ref_canny: torch.Tensor) -> dict[str, torch.Tensor]:
        _, ref_feats = lossy_iframe_ref_bundle(self.iframe_codec, ref_canny)
        return ref_feats

    def _merge_temporal_context(
        self, params: torch.Tensor, ref_feats: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        f4 = self.temporal_ctx_conv(ref_feats["f3"])
        return self.context_merge(torch.cat([params, f4], dim=1))

    def forward(self, batch: dict, training=None):
        if "ref_feats" in batch:
            p_input = self._curr_canny(batch["input"])
            ref_feats = batch["ref_feats"]
        else:
            p_input_gt = batch["input"]
            ref_canny = self._to_1ch(batch["ref_iframe"])
            if training is None:
                training = self.training
            if self.use_lossy_ref:
                _, ref_feats = lossy_iframe_ref_bundle(self.iframe_codec, ref_canny)
            else:
                ref_feats = self._prev_ref_feats(ref_canny)
            p_input = self._curr_canny(p_input_gt)

        if training is None:
            training = self.training
        codec = self.codec

        y = g_a_forward(codec.g_a, p_input, ref_feats, self.fuse)
        z = codec.h_a(y)

        if training:
            z_res = z - codec.means_hyper
            z_hat = codec.ste_round(z_res) + codec.means_hyper
            z_likelihoods = codec.entropy_estimation(codec.add_noise(z_res), codec.scales_hyper)
        else:
            z_res_hat = torch.round(z - codec.means_hyper)
            z_hat = z_res_hat + codec.means_hyper
            z_likelihoods = codec.entropy_estimation(z_res_hat, codec.scales_hyper)

        params = codec.h_s(z_hat)
        params = self._merge_temporal_context(params, ref_feats)
        y_res, y_q, y_hat, scales_y = codec.forward_hpcm(y, params)

        _, x_hat = g_s_decode_multiscale(codec.g_s, y_hat, ref_feats, self.fuse)
        x_hat = x_hat.clamp(0.0, 1.0)

        if training:
            y_likelihoods = codec.entropy_estimation(codec.add_noise(y_res), scales_y)
        else:
            y_res_hat = torch.round(y_res)
            y_likelihoods = codec.entropy_estimation(y_res_hat, scales_y)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @torch.no_grad()
    def compress_p(
        self,
        x_pad: torch.Tensor,
        ref_feats: dict[str, torch.Tensor],
    ) -> dict:
        from src.entropy_models import ubransEncoder

        codec = self.codec
        y = g_a_forward(codec.g_a, x_pad, ref_feats, self.fuse)
        z = codec.h_a(y)
        z_res_hat = torch.round(z - codec.means_hyper)
        indexes_z = codec.build_indexes_z(z_res_hat.size())

        encoder_z = ubransEncoder()
        codec.compress_symbols(
            z_res_hat, indexes_z,
            codec.quantized_cdf_z.cpu().numpy(),
            codec.cdf_length_z.cpu().numpy(),
            codec.offset_z.cpu().numpy(),
            encoder_z,
        )
        z_string = encoder_z.flush()
        z_hat = z_res_hat + codec.means_hyper

        params = self._merge_temporal_context(codec.h_s(z_hat), ref_feats)
        y_q_list, scale_list = codec.forward_hpcm(y, params, write=True)

        encoder_y = ubransEncoder()
        codec.compress_y_two_group_lite(y_q_list, scale_list, encoder_y)
        y_string = encoder_y.flush()
        return {"strings": [y_string, z_string], "shape": z_res_hat.size()[2:]}

    @torch.no_grad()
    def decompress_p(
        self,
        strings,
        shape,
        ref_feats: dict[str, torch.Tensor],
    ) -> dict:
        from src.entropy_models import ubransDecoder

        codec = self.codec
        device = codec.quantized_cdf_z.device
        output_size = (1, codec.scales_hyper.size(1), *shape)
        indexes_z = codec.build_indexes_z(output_size).to(device)

        decoder_z = ubransDecoder()
        decoder_z.set_stream(strings[1])
        z_res_hat = codec.decompress_symbols(
            indexes_z,
            codec.quantized_cdf_z.cpu().numpy(),
            codec.cdf_length_z.cpu().numpy(),
            codec.offset_z.cpu().numpy(),
            decoder_z,
        )
        z_hat = z_res_hat + codec.means_hyper

        params = self._merge_temporal_context(codec.h_s(z_hat), ref_feats)
        decoder_y = ubransDecoder()
        decoder_y.set_stream(strings[0])
        y_hat = codec.decompress_hpcm(params, decoder_y)
        _, x_hat = g_s_decode_multiscale(codec.g_s, y_hat, ref_feats, self.fuse)
        ref_next = ref_feats_from_latent(codec.g_s, y_hat)
        return {
            "x_hat": x_hat.clamp_(0, 1),
            "y_hat": y_hat,
            "ref_feats": ref_next,
        }

    @torch.no_grad()
    def compress_i(self, x_pad: torch.Tensor) -> dict:
        return self.iframe_codec.compress(x_pad)

    @torch.no_grad()
    def decompress_i(self, strings, shape) -> dict:
        out = self.iframe_codec.decompress_with_latent(strings, shape)
        out["ref_feats"] = ref_feats_from_latent(self.iframe_codec.g_s, out["y_hat"])
        return out
