"""P-frame video codec on HPCM_DT1ch: temporal f1-f3 from prev decoder, f4 -> context only."""

from __future__ import annotations

import torch
from torch import nn

from src.layers import conv1x1
from src.utils.distance_transform import canny_to_dt_rgb

from . import HPCM_DT1ch
from .codec_fusion import (
    g_a_forward,
    g_s_decode_multiscale,
    iframe_latent_and_feats,
    ref_feats_from_latent,
)


class HPCM(nn.Module):
    """
    DCVC-style P-frame for DT1ch I-frame codec.

    - iframe_codec: frozen HPCM_DT1ch (I-frame ckpt)
    - codec: trainable HPCM_DT1ch (P-frame main path)
    - f1,f2,f3: multi-scale features from prev-frame decoder (lossy I-frame latent path)
    - f4 = temporal_ctx_conv(f3) merged into h_s params before y entropy (context only)
  """

    LATENT_CH = 320
    CTX_CH = LATENT_CH * 2  # h_s output: scales + means

    def __init__(self, use_lossy_ref: bool = True):
        super().__init__()
        self.use_lossy_ref = use_lossy_ref
        self.iframe_codec = HPCM_DT1ch.HPCM()
        self.codec = HPCM_DT1ch.HPCM()

        self.fuse = nn.ModuleDict({
            "enc_f1": conv1x1(64, 96),
            "enc_f2": conv1x1(128, 192),
            "enc_f3": conv1x1(self.LATENT_CH, 384),
            "dec_f1": conv1x1(64, 64),
            "dec_f2": conv1x1(128, 128),
        })
        self.temporal_ctx_conv = conv1x1(self.LATENT_CH, self.CTX_CH)
        self.context_merge = conv1x1(self.CTX_CH * 2, self.CTX_CH)
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
            for i in range(self.CTX_CH):
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
    def _dist_to_dt_rgb(dist_1ch: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] distance in [0,1] -> [B,3,H,W] DT RGB (on CPU numpy path inside)."""
        out = []
        for b in range(dist_1ch.size(0)):
            out.append(canny_to_dt_rgb(dist_1ch[b, 0]))
        return torch.stack(out, dim=0).to(dist_1ch.device, dist_1ch.dtype)

    def _prev_ref_feats(self, ref_dt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Lossy prev I-frame -> decoder multi-scale f1,f2,f3."""
        _, feats = iframe_latent_and_feats(self.iframe_codec, ref_dt)
        return feats

    def _build_p_input(
        self,
        p_input_gt: torch.Tensor,
        ref_dt: torch.Tensor,
        ref_feats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """R,G from lossy prev DT; B = curr distance (GT)."""
        if not self.use_lossy_ref:
            return p_input_gt
        with torch.no_grad():
            out = self.iframe_codec(ref_dt, training=False)
            prev_dt_hat = self._dist_to_dt_rgb(out["x_hat"])
        curr_b = p_input_gt[:, 2:3]
        return torch.cat([prev_dt_hat[:, 0:1], prev_dt_hat[:, 1:2], curr_b], dim=1)

    def _merge_temporal_context(
        self, params: torch.Tensor, ref_feats: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        f4 = self.temporal_ctx_conv(ref_feats["f3"])
        return self.context_merge(torch.cat([params, f4], dim=1))

    def forward(self, batch: dict, training=None):
        p_input_gt = batch["input"]
        ref_dt = batch["ref_iframe"]
        if training is None:
            training = self.training

        ref_feats = self._prev_ref_feats(ref_dt)
        p_input = self._build_p_input(p_input_gt, ref_dt, ref_feats)
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

    def build_p_input_infer(
        self,
        prev_dist_hat: torch.Tensor,
        curr_canny_1ch: torch.Tensor,
    ) -> torch.Tensor:
        """Build 3ch encoder input for P-frame infer: R,G from prev dist; B from curr."""
        prev_dt = self._dist_to_dt_rgb(prev_dist_hat)
        return torch.cat([prev_dt[:, 0:1], prev_dt[:, 1:2], curr_canny_1ch], dim=1)

    @torch.no_grad()
    def compress_p(
        self,
        x_pad: torch.Tensor,
        ref_feats: dict[str, torch.Tensor],
    ) -> dict:
        """Compress one P-frame; x_pad is padded 3ch DT input."""
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
        """Decompress one P-frame; ref_feats from previous decoded frame."""
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
