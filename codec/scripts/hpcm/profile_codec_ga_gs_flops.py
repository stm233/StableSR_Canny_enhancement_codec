#!/usr/bin/env python3
"""Analytical GFLOPs for g_a / g_s encoder-decoder paths (partial spconv, 256x256).

Counts each layer as dense-equivalent MACs*2 (matches current full-grid spconv runtime).
thop is used only to cross-check dense submodules on CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from thop import profile

CODEC_ROOT = Path(__file__).resolve().parents[2]
STABLESR_ROOT = CODEC_ROOT.parent
sys.path.insert(0, str(STABLESR_ROOT))
sys.path.insert(0, str(CODEC_ROOT))

from src.layers.spconv_conv import _SpconvDenseConvBase
from src.layers.res_blk import PConvRB
from src.layers.spconv_res_blk import SpconvPConvRB


def conv2d_flops(h: int, w: int, k: int, s: int, cin: int, cout: int) -> int:
    ho, wo = h // s, w // s
    return 2 * ho * wo * k * k * cin * cout


def deconv2d_flops(h: int, w: int, k: int, s: int, cin: int, cout: int) -> int:
    ho, wo = h * s, w * s
    return 2 * ho * wo * k * k * cin * cout


def pconvrb_flops(h: int, w: int, n: int, partial_ratio: int = 4, mlp_ratio: int = 4) -> int:
    n1 = n // partial_ratio
    mid = n * mlp_ratio
    macs = h * w * (9 * n1 * n1 + n * mid + mid * n)
    return 2 * macs


def conv1x1_flops(h: int, w: int, cin: int, cout: int) -> int:
    return 2 * h * w * cin * cout


def is_spconv_layer(layer) -> bool:
    if isinstance(layer, (_SpconvDenseConvBase, SpconvPConvRB)):
        return True
    if isinstance(layer, PConvRB):
        return False
    return False


def branch_flops(branch, h: int, w: int, start: int = 0, end: int | None = None) -> tuple[int, int, int]:
    """Return (flops, h_out, w_out) after running branch[start:end]."""
    total = 0
    end = len(branch) if end is None else end
    for i in range(start, end):
        layer = branch[i]
        name = type(layer).__name__
        if "conv4x4" in name or (hasattr(layer, "conv") and "4" in str(getattr(layer, "conv", ""))):
            if "deconv" in name or "Transpose" in name:
                total += deconv2d_flops(h, w, 4, 2, _in_ch(layer), _out_ch(layer))
            else:
                total += conv2d_flops(h, w, 4, 2, _in_ch(layer), _out_ch(layer))
            h, w = h * 2 if "deconv" in name else h // 2, w * 2 if "deconv" in name else w // 2
        elif "conv2x2" in name or "deconv2x2" in name:
            if "deconv" in name:
                total += deconv2d_flops(h, w, 2, 2, _in_ch(layer), _out_ch(layer))
                h, w = h * 2, w * 2
            else:
                total += conv2d_flops(h, w, 2, 2, _in_ch(layer), _out_ch(layer))
                h, w = h // 2, w // 2
        elif isinstance(layer, (PConvRB, SpconvPConvRB)):
            c = _pconvrb_channels(layer)
            total += pconvrb_flops(h, w, c)
        else:
            raise TypeError(f"Unhandled layer {layer} at index {i}")
    return total, h, w


def _in_ch(layer) -> int:
    if isinstance(layer, (_SpconvDenseConvBase,)):
        return layer.conv.in_channels
    if isinstance(layer, torch.nn.Conv2d):
        return layer.in_channels
    if isinstance(layer, torch.nn.ConvTranspose2d):
        return layer.in_channels
    raise TypeError(layer)


def _out_ch(layer) -> int:
    if isinstance(layer, (_SpconvDenseConvBase,)):
        return layer.out_channels
    if isinstance(layer, torch.nn.Conv2d):
        return layer.out_channels
    if isinstance(layer, torch.nn.ConvTranspose2d):
        return layer.out_channels
    raise TypeError(layer)


def _pconvrb_channels(layer) -> int:
    # PConvRB.branch[1] is conv1x1(N, middle); in_channels = N
    return layer.branch[1].in_channels


def ga_branch_flops_manual(h: int, w: int) -> int:
    """Manual walk matching g_a_1ch_partial_spconv topology @ input h,w."""
    flops = 0
    # stage1 spconv: 4x4 s2 -> H/2
    flops += conv2d_flops(h, w, 4, 2, 1, 32)
    h, w = h // 2, w // 2
    flops += 2 * pconvrb_flops(h, w, 32)
    # stage2 spconv: 2x2 s2 -> H/4
    flops += conv2d_flops(h, w, 2, 2, 32, 64)
    h, w = h // 2, w // 2
    flops += 2 * pconvrb_flops(h, w, 64)
    # stage3 dense
    flops += conv2d_flops(h, w, 2, 2, 64, 128)
    h, w = h // 2, w // 2
    flops += 4 * pconvrb_flops(h, w, 128)
    # stage4 dense
    flops += conv2d_flops(h, w, 2, 2, 128, 128)
    return flops


def gs_branch_flops_manual(h: int, w: int) -> int:
    """Manual walk for g_s partial @ latent h,w (16 for 256 input)."""
    flops = 0
    # stage1 dense up
    flops += deconv2d_flops(h, w, 2, 2, 128, 128)
    h, w = h * 2, w * 2
    flops += 4 * pconvrb_flops(h, w, 128)
    # stage2 dense up
    flops += deconv2d_flops(h, w, 2, 2, 128, 64)
    h, w = h * 2, w * 2
    flops += 2 * pconvrb_flops(h, w, 64)
    # stage3 spconv up
    flops += deconv2d_flops(h, w, 2, 2, 64, 32)
    h, w = h * 2, w * 2
    flops += 2 * pconvrb_flops(h, w, 32)
    # stage4 spconv up
    flops += deconv2d_flops(h, w, 4, 2, 32, 1)
    return flops


def cond_extract_flops(h: int, w: int) -> int:
    """cond_g_a: first two stages (layers 0-5), same as g_a prefix."""
    flops = 0
    flops += conv2d_flops(h, w, 4, 2, 1, 32)
    h, w = h // 2, w // 2
    flops += 2 * pconvrb_flops(h, w, 32)
    flops += conv2d_flops(h, w, 2, 2, 32, 64)
    h, w = h // 2, w // 2
    flops += 2 * pconvrb_flops(h, w, 64)
    return flops


def iframe_encoder_flops(h: int, w: int) -> int:
    f = cond_extract_flops(h, w)
    f += ga_branch_flops_manual(h, w)
    f += conv1x1_flops(h // 2, w // 2, 32, 32)  # enc_cond_h2 @ H/2
    f += conv1x1_flops(h // 4, w // 4, 64, 64)  # enc_cond_h4 @ H/4
    return f


def iframe_decoder_flops(h: int, w: int) -> int:
    lh, lw = h // 16, w // 16
    f = gs_branch_flops_manual(lh, lw)
    f += conv1x1_flops(lh * 2, lw * 2, 64, 128)   # dec_cond_h4_f2 @ H/8
    f += conv1x1_flops(lh * 4, lw * 4, 64, 64)    # dec_cond_h4_f1 @ H/4
    f += conv1x1_flops(lh * 8, lw * 8, 32, 32)    # dec_cond_h2 @ H/2
    return f


def pframe_encoder_flops(h: int, w: int) -> int:
    f = iframe_encoder_flops(h, w)
    f += conv1x1_flops(h // 2, w // 2, 64, 32)    # enc_f1 @ H/2
    f += conv1x1_flops(h // 4, w // 4, 128, 64)   # enc_f2 @ H/4
    f += conv1x1_flops(h // 8, w // 8, 128, 128)  # enc_f3 @ H/8
    return f


def pframe_decoder_flops(h: int, w: int) -> int:
    f = iframe_decoder_flops(h, w)
    f += conv1x1_flops(h // 8, w // 8, 128, 128)  # dec_f2 @ H/8
    f += conv1x1_flops(h // 4, w // 4, 64, 64)    # dec_f1 @ H/4
    return f


def thop_verify(name: str, fn, device: str = "cuda") -> float:
    class Wrap(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f
        def forward(self, *args):
            return self.f(*args)
    w = Wrap(fn).to(device).eval()
    # build dummy args inside fn
    try:
        with torch.no_grad():
            macs, _ = profile(w, inputs=(), verbose=False)
        return macs * 2 / 1e9
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    h = w = args.size

    rows = [
        ("I-frame encoder (cond_g_a + g_a + cond fuse)", iframe_encoder_flops(h, w)),
        ("I-frame decoder (g_s + cond fuse)", iframe_decoder_flops(h, w)),
        ("P-frame encoder (+ ref fuse)", pframe_encoder_flops(h, w)),
        ("P-frame decoder (+ ref fuse)", pframe_decoder_flops(h, w)),
    ]

    print(f"Input: {h}x{w} Canny, partial-spconv g_a/g_s")
    print("Method: dense-equivalent MACs×2 (full-grid spconv, matches current runtime)\n")
    print(f"{'Path':<42} {'GFLOPs':>8} {'TFLOPs':>10}")
    print("-" * 62)
    for name, flops in rows:
        print(f"{name:<42} {flops/1e9:8.3f} {flops/1e12:10.6f}")

    # CUDA thop cross-check on dense-only slices
    if torch.cuda.is_available():
        from src.models.codec_fusion import extract_ga_encoder_cond_feats, g_a_forward, g_s_decode_multiscale
        mod = importlib.import_module("src.models.HPCM_Canny1ch_Spconv_Cond")
        st = torch.Tensor(np.exp(np.linspace(np.log(0.12), np.log(64), 60)))
        m = mod.HPCM().eval()
        m.update(st)
        m = m.cuda()
        x = torch.randn(1, 1, h, w, device="cuda")
        cond = torch.randn(1, 1, h, w, device="cuda")
        y = torch.randn(1, 128, h // 16, w // 16, device="cuda")

        def iframe_enc():
            cf = extract_ga_encoder_cond_feats(m.cond_g_a, cond)
            return g_a_forward(m.g_a, x, None, None, cf, m.cond_fuse)

        def iframe_dec():
            cf = extract_ga_encoder_cond_feats(m.cond_g_a, cond)
            return g_s_decode_multiscale(m.g_s, y, None, None, cf, m.cond_fuse)

        pf = importlib.import_module("src.models.HPCM_Video_PFrame_Canny1ch_Spconv_Cond").HPCM().eval().cuda()
        ref = {"f1": torch.randn(1, 64, h // 4, w // 4, device="cuda"),
               "f2": torch.randn(1, 128, h // 8, w // 8, device="cuda"),
               "f3": torch.randn(1, 128, h // 16, w // 16, device="cuda")}

        def pframe_enc():
            cf = extract_ga_encoder_cond_feats(pf.codec.cond_g_a, cond)
            return g_a_forward(pf.codec.g_a, x, ref, pf.fuse, cf, pf.cond_fuse)

        def pframe_dec():
            cf = extract_ga_encoder_cond_feats(pf.codec.cond_g_a, cond)
            return g_s_decode_multiscale(pf.codec.g_s, y, ref, pf.fuse, cf, pf.cond_fuse)

        print("\n--- thop on CUDA (under-counts spconv layers) ---")
        for label, fn in [
            ("I-frame encoder", iframe_enc),
            ("I-frame decoder", iframe_dec),
            ("P-frame encoder", pframe_enc),
            ("P-frame decoder", pframe_dec),
        ]:
            class W(torch.nn.Module):
                def forward(self):
                    fn()
                    return torch.tensor(0.0, device="cuda")
            with torch.no_grad():
                macs, _ = profile(W().cuda(), inputs=(), verbose=False)
            print(f"  {label:<20} {macs*2/1e9:.3f} GFLOPs (thop, spconv≈0)")


if __name__ == "__main__":
    main()
