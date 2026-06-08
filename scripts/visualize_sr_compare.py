#!/usr/bin/env python3
"""
Visual comparison: LQ | GT | Canny | StableSR baseline | your SR (+ optional more).

Row 1: full 512x512 panels (LQ is upscaled to 512 for display).
Row 2+: optional crop zooms (x1,y1,x2,y2 on HR), each patch resized to 512x512.
  Use --crop and/or --crop2, or repeat --crops (0 to many).

Metrics vs GT: PSNR/SSIM (Y channel, same as eval_sr_metrics ycbcr=True) and LPIPS.

Example:
  cd /home/exx/Documents/Tianma/StableSR
  python scripts/visualize_sr_compare.py \
    --lq-dir /data/Dataset/DIV2K/DIV2K_valid_100_512_128/LR_128 \
    --gt-dir /data/Dataset/DIV2K/DIV2K_valid_100_512_128/HR_512 \
    --canny-dir /data/Dataset/DIV2K/DIV2K_valid_100_512_128/canny \
    --baseline-dir /data/Dataset/StableSR-TestSets/outputs_DIV2K_Val100_baseline \
    --mine-dir /data/Dataset/StableSR-TestSets/outputs_DIV2K_Val100_controlnet_canny \
    --mine-label "ControlNet Canny e19" \
    --stem 0801 \
    --crop 200,120,360,280 \
    --crop2 40,300,200,460 \
    --out vis_compare_0801.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.util_image import calculate_ms_ssim, calculate_psnr, calculate_ssim
from taming.modules.losses.lpips import LPIPS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_crop(s: str):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("crop requires x2>x1 and y2>y1")
    return x1, y1, x2, y2


def list_stems(folder: str):
    stems = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f.lower())[1]
        if ext in IMAGE_EXTENSIONS and not f.endswith("_canny.png"):
            stems.append(Path(f).stem)
    return stems


def resolve_image(folder: str, stem: str, suffix: str = ""):
    if not folder:
        return None
    base = Path(folder)
    candidates = [
        base / f"{stem}{suffix}.png",
        base / f"{stem}{suffix}.jpg",
        base / f"{stem}.png",
        base / f"{stem}.jpg",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def load_rgb(path: str, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def crop_resize(img: np.ndarray, box, out_size: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        patch = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    else:
        patch = np.array(
            Image.fromarray(patch).resize((out_size, out_size), Image.LANCZOS),
            dtype=np.uint8,
        )
    return patch


def lq_crop_to_hr_box(lq: np.ndarray, box, hr_size: int, out_size: int) -> np.ndarray:
    """Map HR crop box to LQ coordinates, crop, then upscale to out_size."""
    lh, lw = lq.shape[:2]
    sx, sy = lw / hr_size, lh / hr_size
    x1, y1, x2, y2 = box
    lx1, ly1 = int(x1 * sx), int(y1 * sy)
    lx2, ly2 = max(lx1 + 1, int(round(x2 * sx))), max(ly1 + 1, int(round(y2 * sy)))
    return crop_resize(lq, (lx1, ly1, lx2, ly2), out_size)


def draw_box_outline(img: np.ndarray, box, color=(255, 64, 64), width=2) -> np.ndarray:
    out = img.copy()
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    x1, y1, x2, y2 = box
    for w in range(width):
        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)
    return np.array(pil)


class MetricsComputer:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lpips = LPIPS().eval().to(self.device)

    @staticmethod
    def _align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        if pred.shape == gt.shape:
            return pred
        return np.array(
            Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.LANCZOS),
            dtype=np.uint8,
        )

    def compute(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        pred = self._align(pred, gt)
        pred_f = pred.astype(np.float32)
        gt_f = gt.astype(np.float32)
        psnr_y = calculate_psnr(pred_f, gt_f, border=0, ycbcr=True)
        ssim_y = calculate_ssim(pred_f, gt_f, border=0, ycbcr=True)
        msssim_y = calculate_ms_ssim(pred_f, gt_f, border=0, ycbcr=True)
        pt = lambda a: torch.from_numpy(a / 255.0).permute(2, 0, 1).float().unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lp = self.lpips(pt(pred).to(self.device), pt(gt).to(self.device)).mean().item()
        return {"psnr_y": psnr_y, "ssim_y": ssim_y, "msssim_y": msssim_y, "lpips": lp}


def format_metrics_block(m: dict) -> str:
    return (
        f"PSNR = {m['psnr_y']:.2f}\n"
        f"SSIM = {m['ssim_y']:.4f}\n"
        f"MS-SSIM = {m['msssim_y']:.4f}\n"
        f"LPIPS = {m['lpips']:.4f}"
    )


SR_METRIC_TITLES = frozenset({"StableSR"})


def build_panels(
    stem: str,
    lq_dir: str,
    gt_dir: str,
    canny_dir: str | None,
    baseline_dir: str | None,
    mine_dir: str | None,
    hr_size: int,
    panel_size: int,
):
    lq_path = resolve_image(lq_dir, stem)
    gt_path = resolve_image(gt_dir, stem)
    if lq_path is None or gt_path is None:
        raise FileNotFoundError(f"Missing LQ or GT for stem={stem}")

    lq = load_rgb(lq_path)
    gt = load_rgb(gt_path, size=hr_size)
    lq_vis = np.array(Image.fromarray(lq).resize((hr_size, hr_size), Image.BICUBIC), dtype=np.uint8)

    canny_path = resolve_image(canny_dir, stem) if canny_dir else None
    if canny_path is None and mine_dir:
        canny_path = resolve_image(mine_dir, stem, suffix="_canny")
    if canny_path:
        canny = load_rgb(canny_path, size=hr_size)
        if canny.ndim == 2:
            canny = np.stack([canny] * 3, axis=-1)
    else:
        canny = np.zeros_like(gt)

    baseline_path = resolve_image(baseline_dir, stem) if baseline_dir else None
    mine_path = resolve_image(mine_dir, stem) if mine_dir else None

    panels = [
        ("Input (LQ↑)", lq_vis),
        ("GT", gt),
        ("Canny", canny),
    ]
    paths = {"gt": gt_path}
    if baseline_path:
        panels.append(("StableSR", load_rgb(baseline_path, size=hr_size)))
        paths["baseline"] = baseline_path
    if mine_path:
        panels.append(("Mine", load_rgb(mine_path, size=hr_size)))
        paths["mine"] = mine_path

    # resize all to panel_size for uniform grid
    panels = [(t, crop_resize(im, (0, 0, im.shape[1], im.shape[0]), panel_size)) for t, im in panels]
    return panels, gt, lq, paths


def main():
    parser = argparse.ArgumentParser(description="Visualize SR comparison with crops and metrics")
    parser.add_argument("--lq-dir", type=str, required=True)
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--canny-dir", type=str, default=None, help="precomputed canny PNG folder")
    parser.add_argument("--baseline-dir", type=str, default=None, help="StableSR output folder")
    parser.add_argument("--mine-dir", type=str, default=None, help="your model output folder")
    parser.add_argument("--mine-label", type=str, default="Mine", help="column title for --mine-dir")
    parser.add_argument("--stem", type=str, default=None, help="image id e.g. 0801")
    parser.add_argument("--index", type=int, default=0, help="index in sorted stems if --stem omitted")
    parser.add_argument("--hr-size", type=int, default=512, help="HR spatial size (coords are on this grid)")
    parser.add_argument("--panel-size", type=int, default=512, help="each tile size in the figure")
    parser.add_argument("--crop", type=str, default=None, help="optional crop1: x1,y1,x2,y2 on HR")
    parser.add_argument("--crop2", type=str, default=None, help="optional crop2: x1,y1,x2,y2 on HR")
    parser.add_argument(
        "--crops",
        action="append",
        default=None,
        metavar="X1,Y1,X2,Y2",
        help="extra crop box(es); repeat for 3+ crops, e.g. --crops 10,10,100,100 --crops 200,200,300,300",
    )
    parser.add_argument("--out", type=str, default="vis_compare.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draw-boxes", action="store_true", help="draw crop rectangles on row-1 images")
    opt = parser.parse_args()

    stems = list_stems(opt.gt_dir)
    if not stems:
        raise SystemExit(f"No images in {opt.gt_dir}")
    stem = opt.stem if opt.stem else stems[min(opt.index, len(stems) - 1)]

    panels, gt, lq, paths = build_panels(
        stem,
        opt.lq_dir,
        opt.gt_dir,
        opt.canny_dir,
        opt.baseline_dir,
        opt.mine_dir,
        opt.hr_size,
        opt.panel_size,
    )
    # fix mine column label
    panels = [
        (opt.mine_label if t == "Mine" else t, im) for t, im in panels
    ]

    crop_specs = []
    if opt.crops:
        crop_specs.extend(opt.crops)
    if opt.crop:
        crop_specs.append(opt.crop)
    if opt.crop2:
        crop_specs.append(opt.crop2)
    crops = [(f"Crop {i + 1}", parse_crop(s)) for i, s in enumerate(crop_specs)]

    mc = MetricsComputer(opt.device)
    gt_full = next(im for t, im in panels if t == "GT")

    ncols = len(panels)
    nrows = 1 + len(crops)
    has_sr_metrics = any(t in SR_METRIC_TITLES or t == opt.mine_label for t, _ in panels)
    fig_h = 3.0 * nrows + (1.0 if has_sr_metrics else 0.2)
    fig_w = 3.0 * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Row 0: SR columns — title + metrics above image; others — title only
    for c, (title, im) in enumerate(panels):
        show = im
        if opt.draw_boxes and crops:
            show = im.copy()
            for _, box in crops:
                show = draw_box_outline(show, box)
        axes[0, c].imshow(show)
        if title in SR_METRIC_TITLES or title == opt.mine_label:
            header = f"{title}\n{format_metrics_block(mc.compute(im, gt_full))}"
            axes[0, c].set_title(header, fontsize=9, linespacing=1.35, pad=8)
        else:
            axes[0, c].set_title(title, fontsize=11)
        axes[0, c].axis("off")

    # Crop rows: images only, no titles / metrics
    for r, (_, box) in enumerate(crops, start=1):
        for c, (title, im_full) in enumerate(panels):
            if title.startswith("Input"):
                patch = lq_crop_to_hr_box(lq, box, opt.hr_size, opt.panel_size)
            elif title == "GT":
                patch = crop_resize(gt, box, opt.panel_size)
            elif title == "Canny":
                canny_path = resolve_image(opt.canny_dir, stem) if opt.canny_dir else None
                if canny_path is None and opt.mine_dir:
                    canny_path = resolve_image(opt.mine_dir, stem, suffix="_canny")
                if canny_path:
                    patch = crop_resize(load_rgb(canny_path, size=opt.hr_size), box, opt.panel_size)
                else:
                    patch = np.zeros((opt.panel_size, opt.panel_size, 3), dtype=np.uint8)
            elif title == opt.mine_label and opt.mine_dir:
                p = resolve_image(opt.mine_dir, stem)
                patch = crop_resize(load_rgb(p, size=opt.hr_size), box, opt.panel_size) if p else crop_resize(im_full, box, opt.panel_size)
            elif title == "StableSR" and opt.baseline_dir:
                p = resolve_image(opt.baseline_dir, stem)
                patch = crop_resize(load_rgb(p, size=opt.hr_size), box, opt.panel_size) if p else crop_resize(im_full, box, opt.panel_size)
            else:
                patch = crop_resize(im_full, box, opt.panel_size)
            axes[r, c].imshow(patch)
            axes[r, c].axis("off")

    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    out = Path(opt.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
