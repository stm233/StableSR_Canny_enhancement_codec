#!/usr/bin/env python3
"""Test I-frame HPCM codec on HQ-VSR processed Canny (manifest_iframe.jsonl)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

# Reuse helpers from test.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test import (  # noqa: E402
    AverageMeter,
    _save_results,
    _sync,
    compute_metrics,
    compute_msssim_db,
    crop,
    get_scale_table,
    pad,
    torch2img,
)

def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    """[1,C,H,W] or [C,H,W] in [0,1] -> PIL (L or RGB)."""
    t = x.squeeze(0).detach().cpu().clamp(0, 1).mul(255).round().byte()
    if t.size(0) == 1:
        return Image.fromarray(t.squeeze(0).numpy(), mode="L")
    return Image.fromarray(t.permute(1, 2, 0).numpy(), mode="RGB")


def load_canny_tensor(path: Path) -> torch.Tensor:
    """L-mode Canny PNG -> [1, 1, H, W] float in [0,1]."""
    img = Image.open(path).convert("L")
    t = ToTensor()(img)
    return t.unsqueeze(0)


def load_canny_rgb_tensor(path: Path) -> torch.Tensor:
    """Legacy 3ch load for HPCM_Base checkpoints."""
    img = Image.open(path).convert("L")
    t = ToTensor()(img)
    return t.repeat(3, 1, 1).unsqueeze(0)


def select_records(
    records: list[dict],
    val_only: bool,
    val_ratio: float,
    seed: int,
    max_images: int,
) -> list[dict]:
    if val_only:
        n = len(records)
        val_size = max(1, int(n * val_ratio))
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen).tolist()
        val_idx = set(perm[:val_size])
        records = [records[i] for i in range(n) if i in val_idx]
    if max_images > 0:
        records = records[:max_images]
    return records


def parse_args():
    p = argparse.ArgumentParser(description="Test I-frame codec on HQ-VSR Canny.")
    p.add_argument("--model_name", type=str, default="HPCM_Canny1ch")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--dataset-root",
        type=str,
        default="/data/Dataset/HQ-VSR_processed",
    )
    p.add_argument("--manifest", type=str, default="manifest_iframe.jsonl")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num", type=int, default=60, help="entropy scale table levels")
    p.add_argument("--max-images", type=int, default=0, help="0 = all selected samples")
    p.add_argument(
        "--val-only",
        action="store_true",
        help="Only eval 1%% val split (same seed as train_video.py)",
    )
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="", help="Save visual PNGs under gt/ recon/ compare/")
    p.add_argument("--results_dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use --device cpu.")
    device = torch.device(args.device)

    root = Path(args.dataset_root)
    manifest = root / args.manifest
    records = load_manifest(manifest)
    records = select_records(records, args.val_only, args.val_ratio, args.seed, args.max_images)
    if not records:
        raise RuntimeError("No samples to evaluate.")

    results_dir = args.results_dir or args.outdir
    vis_gt_dir = vis_recon_dir = vis_cmp_dir = None
    if args.outdir:
        vis_gt_dir = os.path.join(args.outdir, "gt")
        vis_recon_dir = os.path.join(args.outdir, "recon")
        vis_cmp_dir = os.path.join(args.outdir, "compare")
        for d in (args.outdir, vis_gt_dir, vis_recon_dir, vis_cmp_dir):
            os.makedirs(d, exist_ok=True)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    import importlib

    net = importlib.import_module(f".{args.model_name}", "src.models").HPCM
    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = net()
    model.eval()
    model.load_state_dict(ckpt, strict=True)
    model.update(get_scale_table(0.12, 64, args.num))
    model = model.to(device)

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    msssim_m = AverageMeter()
    y_bpp_m = AverageMeter()
    z_bpp_m = AverageMeter()
    enc_m = AverageMeter()
    dec_m = AverageMeter()
    per_image = []

    load_fn = load_canny_rgb_tensor
    to_img_gt = lambda t: tensor_to_image(t[:, :1, :, :])
    to_img_recon = tensor_to_image if args.model_name == "HPCM_Canny1ch" else torch2img

    print(f"Samples: {len(records)}  device: {device}")
    for i, rec in enumerate(records):
        canny_path = root / rec["canny"]
        name = Path(rec["canny"]).stem
        x = load_fn(canny_path).to(device)
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, 256)

        _sync(device)
        t0 = time.time()
        with torch.no_grad():
            out_enc = model.compress(x_pad)
        _sync(device)
        enc_t = time.time() - t0

        _sync(device)
        t0 = time.time()
        with torch.no_grad():
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        _sync(device)
        dec_t = time.time() - t0

        x_hat = crop(out_dec["x_hat"], (h, w))

        if args.outdir:
            gt_img = to_img_gt(x)
            recon_img = to_img_recon(x_hat)
            gt_img.save(os.path.join(vis_gt_dir, f"{name}.png"))
            recon_img.save(os.path.join(vis_recon_dir, f"{name}.png"))
            w, h = gt_img.size
            cmp_img = Image.new("RGB", (w * 2, h))
            cmp_img.paste(gt_img, (0, 0))
            cmp_img.paste(recon_img, (w, 0))
            cmp_img.save(os.path.join(vis_cmp_dir, f"{name}.png"))

        x_gt = x[:, :1, :, :] if x_hat.size(1) == 1 and x.size(1) == 3 else x
        psnr = compute_metrics(x_gt, x_hat, 255)["psnr"]
        msssim_db, msssim_metric = compute_msssim_db(x_hat, x_gt, data_range=1.0)
        num_pixels = h * w
        bpp = sum(len(s) for s in out_enc["strings"]) * 8.0 / num_pixels
        ybpp = len(out_enc["strings"][0]) * 8.0 / num_pixels
        zbpp = len(out_enc["strings"][1]) * 8.0 / num_pixels

        bpp_m.update(bpp)
        psnr_m.update(psnr)
        msssim_m.update(msssim_db)
        y_bpp_m.update(ybpp)
        z_bpp_m.update(zbpp)
        enc_m.update(enc_t)
        dec_m.update(dec_t)
        per_image.append({
            "image": name,
            "psnr": float(psnr),
            "msssim_db": float(msssim_db),
            "msssim_metric": msssim_metric,
            "bpp": float(bpp),
            "y_bpp": float(ybpp),
            "z_bpp": float(zbpp),
            "enc_time": float(enc_t),
            "dec_time": float(dec_t),
        })

        if i % 100 == 0:
            print(f"[{i}/{len(records)}] {name}  PSNR={psnr:.2f}  bpp={bpp:.4f}")

    summary = {
        "psnr": float(psnr_m.avg),
        "msssim_db": float(msssim_m.avg),
        "msssim_metric": per_image[0]["msssim_metric"] if per_image else "",
        "bpp": float(bpp_m.avg),
        "y_bpp": float(y_bpp_m.avg),
        "z_bpp": float(z_bpp_m.avg),
        "enc_time": float(enc_m.avg),
        "dec_time": float(dec_m.avg),
    }
    print(
        f"\nHQ-VSR I-frame Test ({len(records)} images):"
        f"\n  PSNR:  {summary['psnr']:.4f}"
        f"\n  MS:    {summary['msssim_db']:.4f}"
        f"\n  bpp:   {summary['bpp']:.6f}"
        f"\n  y bpp: {summary['y_bpp']:.6f}"
        f"\n  z bpp: {summary['z_bpp']:.6f}"
        f"\n  enc:   {summary['enc_time']:.4f}s"
        f"\n  dec:   {summary['dec_time']:.4f}s"
    )

    if results_dir:
        args.dataset = str(root)
        _save_results(results_dir, args, args.checkpoint, per_image, summary)


if __name__ == "__main__":
    main()
