#!/usr/bin/env python3
"""Test I-frame HPCM codec on HQ-VSR processed Canny (manifest_iframe.jsonl)."""

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

from src.utils.distance_transform import (  # noqa: E402
    canny_to_dt_rgb,
    inverted_r_to_edge_uint8,
)

from utils import psnr_continuous

from test import (  # noqa: E402
    AverageMeter,
    _save_results,
    _sync,
    compute_msssim_db,
    crop,
    get_scale_table,
    pad,
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


def dt_visual_images(
    x_hat: torch.Tensor,
    gt_canny: torch.Tensor,
    edge_threshold: float,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """GT binary Canny + recon edge (R_hat>=thr) + continuous inverted R_hat."""
    gt_u8 = (gt_canny.squeeze().detach().cpu().numpy() > 0.5).astype("uint8") * 255
    recon_u8 = inverted_r_to_edge_uint8(x_hat, threshold=edge_threshold)
    r_u8 = (x_hat.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return (
        Image.fromarray(gt_u8, mode="L"),
        Image.fromarray(recon_u8, mode="L"),
        Image.fromarray(r_u8, mode="L"),
    )


def load_canny_tensor(path: Path) -> torch.Tensor:
    """L-mode Canny PNG -> [1, 1, H, W] float in [0,1]."""
    img = Image.open(path).convert("L")
    t = ToTensor()(img)
    return t.unsqueeze(0)


def load_canny_rgb_tensor(path: Path) -> torch.Tensor:
    """R=G=B from L-mode Canny."""
    img = Image.open(path).convert("L")
    t = ToTensor()(img)
    return t.repeat(3, 1, 1).unsqueeze(0)


def prepare_codec_input(
    model_name: str,
    canny_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (codec_input [1,3,H,W], gt_canny [1,1,H,W] float in [0,1])."""
    edge = load_canny_tensor(canny_path)
    if model_name == "HPCM_DT1ch":
        x = canny_to_dt_rgb(edge.squeeze(0)).unsqueeze(0)
    else:
        x = load_canny_rgb_tensor(canny_path)
    return x.to(device), edge.to(device)


def metric_target(model_name: str, x: torch.Tensor, gt_canny: torch.Tensor) -> torch.Tensor:
    """Target for PSNR — same convention as training."""
    if model_name == "HPCM_DT1ch":
        return x[:, :1]
    if model_name in ("HPCM_Canny1ch", "HPCM_Base_Lite"):
        return gt_canny
    return x


def compute_dt_canny_psnr(
    x_hat: torch.Tensor,
    gt_r: torch.Tensor,
    gt_canny: torch.Tensor,
    edge_threshold: float,
) -> tuple[float, float]:
    """Return (PSNR on inverted R map, PSNR on binarized edge Canny)."""
    psnr_dt = psnr_continuous(x_hat, gt_r, peak=255.0).item()

    gt_edge = (gt_canny >= 0.5).float()
    recon_edge = torch.from_numpy(
        inverted_r_to_edge_uint8(x_hat, threshold=edge_threshold).astype("float32") / 255.0
    ).view_as(gt_edge).to(gt_edge.device)
    psnr_canny = psnr_continuous(recon_edge, gt_edge, peak=255.0).item()
    return psnr_dt, psnr_canny


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
    p.add_argument(
        "--edge-threshold",
        type=float,
        default=0.5,
        help="HPCM_DT1ch: R_hat >= threshold -> edge 255 in recon/",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use --device cpu.")
    device = torch.device(args.device)

    root = Path(args.dataset_root)
    records = load_manifest(root / args.manifest)
    records = select_records(records, args.val_only, args.val_ratio, args.seed, args.max_images)
    if not records:
        raise RuntimeError("No samples to evaluate.")

    results_dir = args.results_dir or args.outdir
    if args.outdir:
        for sub in ("gt", "recon", "compare"):
            os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)
        if args.model_name == "HPCM_DT1ch":
            os.makedirs(os.path.join(args.outdir, "recon_r"), exist_ok=True)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    import importlib

    net = importlib.import_module(f".{args.model_name}", "src.models").HPCM
    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = net().eval()
    model.load_state_dict(ckpt, strict=True)
    model.update(get_scale_table(0.12, 64, args.num))
    model = model.to(device)

    metric_desc = (
        "inverted R PSNR + binarized edge PSNR (R_hat>=thr)"
        if args.model_name == "HPCM_DT1ch"
        else "continuous [0,1] Canny PSNR"
    )
    print(
        f"Samples: {len(records)}  device: {device}  model: {args.model_name}\n"
        f"Metrics: {metric_desc}"
    )

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    psnr_dt_m = AverageMeter()
    psnr_canny_m = AverageMeter()
    msssim_m = AverageMeter()
    y_bpp_m = AverageMeter()
    z_bpp_m = AverageMeter()
    enc_m = AverageMeter()
    dec_m = AverageMeter()
    per_image = []

    for i, rec in enumerate(records):
        canny_path = root / rec["canny"]
        name = Path(rec["canny"]).stem
        x, gt_canny = prepare_codec_input(args.model_name, canny_path, device)
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
        target = metric_target(args.model_name, x, gt_canny)

        if args.model_name == "HPCM_DT1ch":
            psnr_dt, psnr_canny = compute_dt_canny_psnr(
                x_hat, target, gt_canny, args.edge_threshold
            )
            psnr = psnr_dt
            psnr_dt_m.update(psnr_dt)
            psnr_canny_m.update(psnr_canny)
        else:
            psnr = psnr_continuous(x_hat, target, peak=255.0).item()
            psnr_canny = psnr
        msssim_db, msssim_metric = compute_msssim_db(x_hat, target, data_range=1.0)

        if args.outdir:
            if args.model_name == "HPCM_DT1ch":
                gt_img, recon_img, r_img = dt_visual_images(
                    x_hat, gt_canny, args.edge_threshold
                )
                r_img.save(os.path.join(args.outdir, "recon_r", f"{name}.png"))
            else:
                gt_img = tensor_to_image(target)
                recon_img = tensor_to_image(x_hat)
            gt_img.save(os.path.join(args.outdir, "gt", f"{name}.png"))
            recon_img.save(os.path.join(args.outdir, "recon", f"{name}.png"))
            cmp_img = Image.new("RGB", (w * 2, h))
            cmp_img.paste(gt_img.convert("RGB"), (0, 0))
            cmp_img.paste(recon_img.convert("RGB"), (w, 0))
            cmp_img.save(os.path.join(args.outdir, "compare", f"{name}.png"))

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
            "psnr_dt": float(psnr_dt) if args.model_name == "HPCM_DT1ch" else float(psnr),
            "psnr_canny": float(psnr_canny),
            "msssim_db": float(msssim_db),
            "msssim_metric": msssim_metric,
            "bpp": float(bpp),
            "y_bpp": float(ybpp),
            "z_bpp": float(zbpp),
            "enc_time": float(enc_t),
            "dec_time": float(dec_t),
        })

        if i % 100 == 0:
            if args.model_name == "HPCM_DT1ch":
                print(
                    f"[{i}/{len(records)}] {name}  "
                    f"PSNR_DT={psnr_dt:.2f}  PSNR_canny={psnr_canny:.2f}  bpp={bpp:.4f}"
                )
            else:
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
    if args.model_name == "HPCM_DT1ch":
        summary["psnr_dt"] = float(psnr_dt_m.avg)
        summary["psnr_canny"] = float(psnr_canny_m.avg)
        psnr_lines = (
            f"\n  PSNR (DT / inverted R): {summary['psnr_dt']:.4f}"
            f"\n  PSNR (Canny edge):      {summary['psnr_canny']:.4f}"
        )
    else:
        psnr_lines = f"\n  PSNR:  {summary['psnr']:.4f}"
    print(
        f"\nHQ-VSR I-frame Test ({len(records)} images):"
        f"{psnr_lines}"
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
