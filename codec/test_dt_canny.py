#!/usr/bin/env python3
"""Test HPCM_DT1ch: DT 3ch in, decode Canny directly."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test import (  # noqa: E402
    AverageMeter,
    compute_metrics,
    crop,
    get_scale_table,
    pad,
)

from src.utils.distance_transform import canny_to_dt_rgb  # noqa: E402


def load_canny_l(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    t = ToTensor()(img)
    return t.unsqueeze(0)


def parse_args():
    p = argparse.ArgumentParser(description="Test DT Canny codec.")
    p.add_argument("--model_name", type=str, default="HPCM_DT1ch")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, help="Dir of binary Canny PNGs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num", type=int, default=60)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--outdir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    root = Path(args.dataset)
    files = sorted(
        f for f in root.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if args.max_images > 0:
        files = files[: args.max_images]

    import importlib

    net = importlib.import_module(f".{args.model_name}", "src.models").HPCM
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = net().eval()
    model.load_state_dict(ckpt, strict=True)
    model.update(get_scale_table(0.12, 64, args.num))
    model = model.to(device)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        for sub in ("gt_edge", "recon_edge", "compare"):
            os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()

    for i, fpath in enumerate(files):
        edge = load_canny_l(fpath)
        h, w = edge.size(2), edge.size(3)
        x = canny_to_dt_rgb(edge.squeeze(0)).unsqueeze(0).to(device)
        x_pad = pad(x, 256)

        with torch.no_grad():
            enc = model.compress(x_pad)
            dec = model.decompress(enc["strings"], enc["shape"])
        canny_hat = crop(dec["x_hat"], (h, w))

        gt_edge_u8 = (edge.squeeze().numpy() > 0.5).astype(np.uint8) * 255
        recon_edge_u8 = (canny_hat.squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255

        gt_t = torch.from_numpy(gt_edge_u8.astype(np.float32) / 255.0).view(1, 1, h, w)
        rec_t = torch.from_numpy(recon_edge_u8.astype(np.float32) / 255.0).view(1, 1, h, w)
        psnr = compute_metrics(gt_t, rec_t, 255)["psnr"]

        num_pixels = h * w
        bpp = sum(len(s) for s in enc["strings"]) * 8.0 / num_pixels
        bpp_m.update(bpp)
        psnr_m.update(psnr)

        name = fpath.stem
        if args.outdir:
            Image.fromarray(gt_edge_u8, mode="L").save(
                os.path.join(args.outdir, "gt_edge", f"{name}.png")
            )
            Image.fromarray(recon_edge_u8, mode="L").save(
                os.path.join(args.outdir, "recon_edge", f"{name}.png")
            )
            cmp = Image.new("RGB", (w * 2, h))
            cmp.paste(Image.fromarray(gt_edge_u8, mode="L").convert("RGB"), (0, 0))
            cmp.paste(Image.fromarray(recon_edge_u8, mode="L").convert("RGB"), (w, 0))
            cmp.save(os.path.join(args.outdir, "compare", f"{name}.png"))

        if i % 50 == 0:
            print(f"[{i}/{len(files)}] {name}  PSNR={psnr:.2f}  bpp={bpp:.4f}")

    print(f"\nDT codec test ({len(files)} images)")
    print(f"  PSNR (canny): {psnr_m.avg:.4f}")
    print(f"  bpp:          {bpp_m.avg:.6f}")


if __name__ == "__main__":
    main()
