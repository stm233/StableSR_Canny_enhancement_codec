#!/usr/bin/env python3
"""Test P-frame HPCM_Video_PFrame on HQ-VSR test manifest (consecutive pairs)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test import AverageMeter, _save_results, _sync, compute_metrics, compute_msssim_db  # noqa: E402


def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def l_to_tensor(path: Path) -> torch.Tensor:
    t = ToTensor()(Image.open(path).convert("L"))
    return t.unsqueeze(0)


def load_pframe_batch(root: Path, rec: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    prev = l_to_tensor(root / rec["prev_canny"])
    curr = l_to_tensor(root / rec["curr_canny"])
    p_input = torch.cat([prev, prev, curr], dim=0)
    ref_iframe = prev.repeat(3, 1, 1)
    target = curr
    model_batch = {
        "input": p_input.unsqueeze(0).to(device),
        "ref_iframe": ref_iframe.unsqueeze(0).to(device),
    }
    return model_batch, target.unsqueeze(0).to(device)


def parse_args():
    p = argparse.ArgumentParser(description="Test P-frame codec on HQ-VSR test manifest.")
    p.add_argument("--checkpoint", type=str, required=True, help="Full HPCM_Video_PFrame ckpt")
    p.add_argument("--iframe-checkpoint", type=str, default="", help="Optional if ckpt lacks iframe weights")
    p.add_argument("--dataset-root", type=str, default="/data/Dataset/HQ-VSR_test500")
    p.add_argument("--manifest", type=str, default="manifest_pframe.jsonl")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--results_dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    root = Path(args.dataset_root)
    records = load_manifest(root / args.manifest)
    if args.max_images > 0:
        records = records[: args.max_images]

    from src.models.HPCM_Video_PFrame import HPCM

    model = HPCM(use_lossy_ref=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    for p in model.iframe_codec.parameters():
        p.requires_grad = False
    model.iframe_codec.eval()
    model.eval()

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    msssim_m = AverageMeter()
    per_image = []

    print(f"P-frame test: {len(records)} pairs, device={device}")
    for i, rec in enumerate(records):
        batch, target = load_pframe_batch(root, rec, device)
        _sync(device)
        t0 = time.time()
        with torch.no_grad():
            out = model(batch, training=False)
        _sync(device)
        dt = time.time() - t0

        x_hat = out["x_hat"]
        n, _, h, w = target.shape
        num_pixels = n * h * w
        bpp = sum(
            torch.log(lk).sum().item() / (-torch.log(torch.tensor(2.0)).item() * num_pixels)
            for lk in out["likelihoods"].values()
        )
        org = (target * 255).clamp(0, 255)
        rec_t = (x_hat * 255).clamp(0, 255)
        psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10((org - rec_t).pow(2).mean())
        msssim_db, msssim_metric = compute_msssim_db(x_hat, target, data_range=1.0)

        name = rec["curr_canny"].replace("/", "_")
        bpp_m.update(bpp)
        psnr_m.update(psnr.item())
        msssim_m.update(msssim_db)
        per_image.append({
            "image": name,
            "psnr": float(psnr.item()),
            "msssim_db": float(msssim_db),
            "msssim_metric": msssim_metric,
            "bpp": float(bpp),
            "y_bpp": float(bpp),
            "z_bpp": 0.0,
            "enc_time": float(dt),
            "dec_time": 0.0,
        })
        if i % 50 == 0:
            print(f"[{i}/{len(records)}] {name}  PSNR={psnr.item():.2f}  est_bpp={bpp:.4f}")

    summary = {
        "psnr": float(psnr_m.avg),
        "msssim_db": float(msssim_m.avg),
        "msssim_metric": per_image[0]["msssim_metric"] if per_image else "",
        "bpp": float(bpp_m.avg),
        "y_bpp": float(bpp_m.avg),
        "z_bpp": 0.0,
        "enc_time": float(sum(r["enc_time"] for r in per_image) / max(len(per_image), 1)),
        "dec_time": 0.0,
        "note": "bpp from likelihoods (forward), not bitstream compress yet",
    }
    print(f"\nP-frame summary: PSNR={summary['psnr']:.4f}  est_bpp={summary['bpp']:.6f}")
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        args.model_name = "HPCM_Video_PFrame"
        args.dataset = str(root)
        args.outdir = ""
        _save_results(args.results_dir, args, args.checkpoint, per_image, summary)


if __name__ == "__main__":
    main()
