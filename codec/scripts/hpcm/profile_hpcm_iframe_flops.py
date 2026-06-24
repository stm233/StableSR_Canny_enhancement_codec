#!/usr/bin/env python3
"""Profile GFLOPs for HPCM I-frame models at 512x512."""

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


def get_scale_table(min_val: float, max_val: float, levels: int) -> torch.Tensor:
    return torch.Tensor(np.exp(np.linspace(np.log(min_val), np.log(max_val), levels)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="HPCM_Canny1ch",
        choices=["HPCM_Base", "HPCM_Base_Lite", "HPCM_Canny1ch", "HPCM_DT1ch"],
    )
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    mod = importlib.import_module(f"src.models.{args.model}")
    model = mod.HPCM().eval()
    model.update(get_scale_table(0.12, 64, 60))

    in_ch = 1 if args.model == "HPCM_Canny1ch" else 3
    x = torch.randn(1, in_ch, args.size, args.size)
    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)
    gflops = macs * 2 / 1e9

    parts = {
        "g_a": (model.g_a, x),
        "g_s": (model.g_s, model.g_a(x)),
        "h_a": (model.h_a, model.g_a(x)),
    }
    z_hat = torch.round(model.h_a(model.g_a(x)) - model.means_hyper) + model.means_hyper
    parts["h_s"] = (model.h_s, z_hat)

    print(f"Model:  {args.model}")
    print(f"Input:  {args.size}x{args.size}")
    print(f"Params: {int(params):,}")
    print(f"Total:  {gflops:.2f} GFLOPs")
    print("--- submodule (single pass, excludes entropy loops) ---")
    for name, (m, inp) in parts.items():
        with torch.no_grad():
            m_macs, _ = profile(m, inputs=(inp,), verbose=False)
        print(f"  {name:6s} {m_macs * 2 / 1e9:8.2f} GFLOPs")


if __name__ == "__main__":
    main()
