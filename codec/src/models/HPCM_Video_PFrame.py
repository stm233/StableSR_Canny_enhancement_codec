"""P-frame video codec: I-frame ckpt only for ref path; separate P-frame codec.

DCVC-style training: previous I-frame is passed through frozen iframe_codec (lossy)
before feeding ref_encoder and P-frame encoder R/G channels.

- iframe_codec: frozen, loaded from your I-frame / canny checkpoint
- ref_encoder: trainable g_a, init from iframe_codec.g_a
- codec: separate P-frame main codec (NOT loaded from I-frame ckpt)
- dec_fuse + canny_head: trainable decoder fusion
"""

from __future__ import annotations

import torch
from torch import nn

from .HPCM_Base import HPCM as BaseHPCM
from .HPCM_Base import g_a
from src.layers import conv1x1


class HPCM(nn.Module):
    """P-frame model with frozen I-frame codec for lossy reference (DCVC-style)."""

    LATENT_CH = 320

    def __init__(self, use_lossy_ref: bool = True):
        super().__init__()
        self.use_lossy_ref = use_lossy_ref
        self.iframe_codec = BaseHPCM()
        self.ref_encoder = g_a()
        self.codec = BaseHPCM()
        self.dec_fuse = conv1x1(self.LATENT_CH * 2, self.LATENT_CH)
        self.canny_head = nn.Conv2d(3, 1, kernel_size=1)
        self._init_fusion_heads()

    def _init_fusion_heads(self) -> None:
        with torch.no_grad():
            self.dec_fuse.weight.zero_()
            self.dec_fuse.bias.zero_()
            for i in range(self.LATENT_CH):
                self.dec_fuse.weight[i, i, 0, 0] = 1.0

            self.canny_head.weight.zero_()
            self.canny_head.bias.zero_()
            self.canny_head.weight[0, 2, 0, 0] = 1.0

    def load_iframe_checkpoint(self, path: str, map_location=None) -> None:
        """Load I-frame / canny ckpt into frozen iframe_codec; init ref_encoder from its g_a.

        Does NOT load into P-frame self.codec.
        """
        ckpt = torch.load(path, map_location=map_location)
        self.iframe_codec.load_state_dict(ckpt, strict=True)
        for param in self.iframe_codec.parameters():
            param.requires_grad = False
        self.iframe_codec.eval()
        self.ref_encoder.load_state_dict(self.iframe_codec.g_a.state_dict())

    def load_p_codec_checkpoint(self, path: str, map_location=None) -> None:
        """Optional: initialize P-frame main codec from a separate checkpoint."""
        ckpt = torch.load(path, map_location=map_location)
        self.codec.load_state_dict(ckpt, strict=True)

    def _iframe_reconstruct(self, ref_iframe: torch.Tensor) -> torch.Tensor:
        """Lossy I-frame round-trip via frozen iframe_codec (no grad)."""
        with torch.no_grad():
            out = self.iframe_codec(ref_iframe, training=False)
            return out["x_hat"].clamp(0.0, 1.0)

    def load_resume_checkpoint(self, path: str, map_location=None) -> None:
        """Load full P-frame model (resume training or inference)."""
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt, strict=True)
        for param in self.iframe_codec.parameters():
            param.requires_grad = False
        self.iframe_codec.eval()

    def forward(self, batch: dict, training=None):
        p_input_gt = batch["input"]
        ref_iframe_gt = batch["ref_iframe"]
        if training is None:
            training = self.training

        if self.use_lossy_ref:
            ref_iframe = self._iframe_reconstruct(ref_iframe_gt)
            prev_hat_1ch = ref_iframe.mean(dim=1, keepdim=True)
            curr_canny_1ch = p_input_gt[:, 2:3]
            p_input = torch.cat([prev_hat_1ch, prev_hat_1ch, curr_canny_1ch], dim=1)
        else:
            ref_iframe = ref_iframe_gt
            p_input = p_input_gt

        f_ref = self.ref_encoder(ref_iframe)
        codec = self.codec

        y = codec.g_a(p_input)
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
        y_res, y_q, y_hat, scales_y = codec.forward_hpcm(
            y,
            params,
            codec.y_spatial_prior_adaptor_list_s1,
            codec.y_spatial_prior_s1_s2,
            codec.y_spatial_prior_adaptor_list_s2,
            codec.y_spatial_prior_s1_s2,
            codec.y_spatial_prior_adaptor_list_s3,
            codec.y_spatial_prior_s3,
            codec.adaptive_params_list,
            codec.context_net,
        )

        fused = self.dec_fuse(torch.cat([y_hat, f_ref], dim=1))
        x_hat_rgb = codec.g_s(fused)
        canny_hat = self.canny_head(x_hat_rgb).clamp(0.0, 1.0)

        if training:
            y_likelihoods = codec.entropy_estimation(codec.add_noise(y_res), scales_y)
        else:
            y_res_hat = torch.round(y_res)
            y_likelihoods = codec.entropy_estimation(y_res_hat, scales_y)

        return {
            "x_hat": canny_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
