"""Multi-scale fusion helpers for P-frame DT1ch codec."""

from __future__ import annotations

import torch
from torch import nn

from src.layers import conv1x1


def _fuse(x: torch.Tensor, ref: torch.Tensor, proj: nn.Module) -> torch.Tensor:
    if ref.shape[-2:] != x.shape[-2:]:
        ref = nn.functional.interpolate(
            ref, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
    return x + proj(ref)


def extract_ga_encoder_cond_feats(g_a: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Partial g_a forward: features at H/2 (32ch) and H/4 (64ch)."""
    b = g_a.branch
    x = b[0](x)
    x = b[1](x)
    x = b[2](x)
    cond_h2 = x
    x = b[3](x)
    x = b[4](x)
    x = b[5](x)
    cond_h4 = x
    return {"cond_h2": cond_h2, "cond_h4": cond_h4}


def g_a_forward(
    g_a: nn.Module,
    x: torch.Tensor,
    ref: dict[str, torch.Tensor] | None,
    fuse: nn.ModuleDict | None,
    cond_feats: dict[str, torch.Tensor] | None = None,
    cond_fuse: nn.ModuleDict | None = None,
) -> torch.Tensor:
    """g_a with optional decoder ref features f1@H/4, f2@H/8, f3@H/16 and cond @ H/2,H/4."""
    b = g_a.branch
    x = b[0](x)
    x = b[1](x)
    x = b[2](x)
    if cond_feats is not None and cond_fuse is not None and "cond_h2" in cond_feats:
        x = _fuse(x, cond_feats["cond_h2"], cond_fuse["enc_cond_h2"])
    if ref is not None and fuse is not None and "f1" in ref:
        x = _fuse(x, ref["f1"], fuse["enc_f1"])

    x = b[3](x)
    x = b[4](x)
    x = b[5](x)
    if cond_feats is not None and cond_fuse is not None and "cond_h4" in cond_feats:
        x = _fuse(x, cond_feats["cond_h4"], cond_fuse["enc_cond_h4"])
    if ref is not None and fuse is not None and "f2" in ref:
        x = _fuse(x, ref["f2"], fuse["enc_f2"])

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
    cond_feats: dict[str, torch.Tensor] | None = None,
    cond_fuse: nn.ModuleDict | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Decode latent y -> x_hat; return decoder features for next frame.

    f3: latent input @ H/16 (320ch)
    f2: @ H/8  (128ch)
    f1: @ H/4  (64ch)
    """
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
    if cond_feats is not None and cond_fuse is not None and "cond_h4" in cond_feats:
        x = _fuse(x, cond_feats["cond_h4"], cond_fuse["dec_cond_h4_f2"])

    x = b[5](x)
    x = b[6](x)
    x = b[7](x)
    feats["f1"] = x
    if ref is not None and fuse is not None and "f1" in ref:
        x = _fuse(x, ref["f1"], fuse["dec_f1"])
    if cond_feats is not None and cond_fuse is not None and "cond_h4" in cond_feats:
        x = _fuse(x, cond_feats["cond_h4"], cond_fuse["dec_cond_h4_f1"])

    x = b[8](x)
    x = b[9](x)
    x = b[10](x)
    if cond_feats is not None and cond_fuse is not None and "cond_h2" in cond_feats:
        x = _fuse(x, cond_feats["cond_h2"], cond_fuse["dec_cond_h2"])
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
