"""Multi-scale fusion helpers for P-frame DT1ch codec."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.layers import conv1x1


def _fuse(x: torch.Tensor, ref: torch.Tensor, proj: nn.Module) -> torch.Tensor:
    if ref.shape[-2:] != x.shape[-2:]:
        ref = F.interpolate(
            ref, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
    return x + proj(ref)


def _align_cond(cond: torch.Tensor | None, hw: tuple[int, int]) -> torch.Tensor | None:
    if cond is None:
        return None
    if cond.shape[-2:] != hw:
        cond = F.interpolate(cond, size=hw, mode="bilinear", align_corners=False)
    if cond.shape[1] > 1:
        cond = cond[:, :1]
    return cond


def _concat_cond(feat: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
    if cond is None:
        zeros = torch.zeros(
            feat.size(0), 1, feat.size(2), feat.size(3),
            dtype=feat.dtype, device=feat.device,
        )
        return torch.cat([feat, zeros], dim=1)
    cond = _align_cond(cond, feat.shape[-2:])
    return torch.cat([feat, cond], dim=1)


def g_a_forward(
    g_a: nn.Module,
    x: torch.Tensor,
    ref: dict[str, torch.Tensor] | None,
    fuse: nn.ModuleDict | None,
    cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """g_a: sparse stages to H/4, concat cond @ 64x64, dense stages to latent."""
    b = g_a.branch
    x = b[0](x)
    x = b[1](x)
    x = b[2](x)
    if ref is not None and fuse is not None and "f1" in ref:
        x = _fuse(x, ref["f1"], fuse["enc_f1"])

    x = b[3](x)
    x = b[4](x)
    x = b[5](x)
    if ref is not None and fuse is not None and "f2" in ref:
        x = _fuse(x, ref["f2"], fuse["enc_f2"])
    x = _concat_cond(x, cond)

    x = b[6](x)
    x = b[7](x)
    x = b[8](x)
    x = b[9](x)
    x = b[10](x)
    if ref is not None and fuse is not None and "f3" in ref:
        x = _fuse(x, ref["f3"], fuse["enc_f3"])

    return b[11](x)


def g_s_decode_multiscale(
    g_s: nn.Module,
    y: torch.Tensor,
    ref: dict[str, torch.Tensor] | None = None,
    fuse: nn.ModuleDict | None = None,
    cond: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Decode latent y -> x_hat; concat cond @ 64x64 symmetric to encoder."""
    b = g_s.branch
    feats = {"f3": y}

    x = b[0](y)
    x = b[1](x)
    x = b[2](x)
    x = b[3](x)
    x = b[4](x)
    feats["f2"] = x
    if ref is not None and fuse is not None and "f2" in ref:
        x = _fuse(x, ref["f2"], fuse["dec_f2"])

    x = b[5](x)
    x = b[6](x)
    x = b[7](x)
    feats["f1"] = x
    if ref is not None and fuse is not None and "f1" in ref:
        x = _fuse(x, ref["f1"], fuse["dec_f1"])
    x = _concat_cond(x, cond)

    x = b[8](x)
    x = b[9](x)
    x = b[10](x)
    x = b[11](x)
    return feats, x


def iframe_latent_and_feats(
    codec: nn.Module,
    x_dt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Lossy I-frame round-trip: DT RGB in -> y_hat + decoder f1/f2/f3 (no grad)."""
    with torch.no_grad():
        y = g_a_forward(codec.g_a, x_dt, None, None)
        z = codec.h_a(y)
        z_res_hat = torch.round(z - codec.means_hyper)
        z_hat = z_res_hat + codec.means_hyper
        params = codec.h_s(z_hat)
        _, _, y_hat, _ = codec.forward_hpcm(y, params, write=False)
        feats, _ = g_s_decode_multiscale(codec.g_s, y_hat, None, None)
    return y_hat, feats


def lossy_iframe_ref_bundle(
    codec: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single lossy I-frame pass: x_hat + decoder f1/f2/f3. x is 1ch (Canny1ch) or 3ch (DT)."""
    with torch.no_grad():
        y = g_a_forward(codec.g_a, x, None, None)
        z = codec.h_a(y)
        z_res_hat = torch.round(z - codec.means_hyper)
        z_hat = z_res_hat + codec.means_hyper
        params = codec.h_s(z_hat)
        _, _, y_hat, _ = codec.forward_hpcm(y, params, write=False)
        feats, x_hat = g_s_decode_multiscale(codec.g_s, y_hat, None, None)
    return x_hat, feats


def ref_feats_from_latent(
    g_s: nn.Module,
    y_hat: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Decoder multi-scale features f1/f2/f3 from latent (no temporal fusion)."""
    feats, _ = g_s_decode_multiscale(g_s, y_hat, None, None)
    return feats
