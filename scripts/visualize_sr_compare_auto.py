#!/usr/bin/env python3
"""
Auto-pick best showcase images & crop regions, then render same layout as visualize_sr_compare.py.

1) Full-image PSNR(Y) gap = mine - baseline vs GT; keep top-K images (default 10).
2) On each selected image, slide a crop_size x crop_size window (step crop_step) on HR grid;
   pick top-N crop boxes with largest PSNR gap (default 2, non-overlapping when possible).
3) Save one PNG per image under --out-dir plus summary.txt.

Example:
  python scripts/visualize_sr_compare_auto.py \\
    --lq-dir /data/Dataset/DIV2K/DIV2K_valid_100_512_128/LR_128 \\
    --gt-dir /data/Dataset/DIV2K/DIV2K_valid_100_512_128/HR_512 \\
    --canny-dir /data/Dataset/DIV2K/DIV2K_valid_100_512_128/canny \\
    --baseline-dir /data/Dataset/StableSR-TestSets/outputs_DIV2K_Val100_baseline2 \\
    --mine-dir /data/Dataset/StableSR-TestSets/outputs_DIV2K_Val100_controlnet_canny_20_origcfw \\
    --mine-label "CN Canny e20" \\
    --out-dir /data/Dataset/StableSR-TestSets/vis_auto_top10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.util_image import calculate_psnr
from scripts.visualize_sr_compare import (
    SR_METRIC_TITLES,
    MetricsComputer,
    build_panels,
    crop_resize,
    draw_box_outline,
    format_metrics_block,
    list_stems,
    load_rgb,
    lq_crop_to_hr_box,
    resolve_image,
)


def psnr_y(pred: np.ndarray, gt: np.ndarray) -> float:
    return calculate_psnr(pred.astype(np.float32), gt.astype(np.float32), border=0, ycbcr=True)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def sliding_boxes(h: int, w: int, size: int, step: int):
    if size > h or size > w:
        return [(0, 0, w, h)]
    boxes = []
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            boxes.append((x, y, x + size, y + size))
    return boxes


def pick_top_boxes_by_gap(
    gt: np.ndarray,
    baseline: np.ndarray,
    mine: np.ndarray,
    crop_size: int,
    crop_step: int,
    top_n: int,
    max_iou: float = 0.25,
):
    h, w = gt.shape[:2]
    candidates = []
    for box in sliding_boxes(h, w, crop_size, crop_step):
        g = crop_resize(gt, box, crop_size)
        b = crop_resize(baseline, box, crop_size)
        m = crop_resize(mine, box, crop_size)
        gap = psnr_y(m, g) - psnr_y(b, g)
        candidates.append((gap, box))

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = []
    for gap, box in candidates:
        if len(picked) >= top_n:
            break
        if all(box_iou(box, pb) <= max_iou for _, pb in picked):
            picked.append((gap, box))
    # fallback: allow overlap if not enough distinct regions
    for gap, box in candidates:
        if len(picked) >= top_n:
            break
        if box not in [b for _, b in picked]:
            picked.append((gap, box))
    return picked[:top_n]


def rank_images_by_full_psnr_gap(
    stems,
    gt_dir: str,
    baseline_dir: str,
    mine_dir: str,
    hr_size: int,
):
    rows = []
    for stem in stems:
        gt_path = resolve_image(gt_dir, stem)
        base_path = resolve_image(baseline_dir, stem)
        mine_path = resolve_image(mine_dir, stem)
        if not all([gt_path, base_path, mine_path]):
            continue
        gt = load_rgb(gt_path, size=hr_size)
        base = load_rgb(base_path, size=hr_size)
        mine = load_rgb(mine_path, size=hr_size)
        p_base = psnr_y(base, gt)
        p_mine = psnr_y(mine, gt)
        rows.append({
            "stem": stem,
            "psnr_base": p_base,
            "psnr_mine": p_mine,
            "gap": p_mine - p_base,
        })
    rows.sort(key=lambda r: r["gap"], reverse=True)
    return rows


def render_comparison(
    stem: str,
    panels,
    gt: np.ndarray,
    lq: np.ndarray,
    crops,
    mine_label: str,
    canny_dir: str | None,
    baseline_dir: str | None,
    mine_dir: str | None,
    hr_size: int,
    panel_size: int,
    mc: MetricsComputer,
    draw_boxes: bool,
    out_path: Path,
):
    gt_full = next(im for t, im in panels if t == "GT")
    ncols = len(panels)
    nrows = 1 + len(crops)
    has_sr_metrics = any(t in SR_METRIC_TITLES or t == mine_label for t, _ in panels)
    fig_h = 3.0 * nrows + (1.0 if has_sr_metrics else 0.2)
    fig_w = 3.0 * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for c, (title, im) in enumerate(panels):
        show = im
        if draw_boxes and crops:
            show = im.copy()
            for _, box in crops:
                show = draw_box_outline(show, box)
        axes[0, c].imshow(show)
        if title in SR_METRIC_TITLES or title == mine_label:
            header = f"{title}\n{format_metrics_block(mc.compute(im, gt_full))}"
            axes[0, c].set_title(header, fontsize=9, linespacing=1.35, pad=8)
        else:
            axes[0, c].set_title(title, fontsize=11)
        axes[0, c].axis("off")

    for r, (_, box) in enumerate(crops, start=1):
        for c, (title, im_full) in enumerate(panels):
            if title.startswith("Input"):
                patch = lq_crop_to_hr_box(lq, box, hr_size, panel_size)
            elif title == "GT":
                patch = crop_resize(gt, box, panel_size)
            elif title == "Canny":
                canny_path = resolve_image(canny_dir, stem) if canny_dir else None
                if canny_path is None and mine_dir:
                    canny_path = resolve_image(mine_dir, stem, suffix="_canny")
                if canny_path:
                    patch = crop_resize(load_rgb(canny_path, size=hr_size), box, panel_size)
                else:
                    patch = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
            elif title == mine_label and mine_dir:
                p = resolve_image(mine_dir, stem)
                patch = crop_resize(load_rgb(p, size=hr_size), box, panel_size) if p else crop_resize(im_full, box, panel_size)
            elif title == "StableSR" and baseline_dir:
                p = resolve_image(baseline_dir, stem)
                patch = crop_resize(load_rgb(p, size=hr_size), box, panel_size) if p else crop_resize(im_full, box, panel_size)
            else:
                patch = crop_resize(im_full, box, panel_size)
            axes[r, c].imshow(patch)
            axes[r, c].axis("off")

    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Auto top-gap SR visual comparison")
    parser.add_argument("--lq-dir", type=str, required=True)
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--mine-dir", type=str, required=True)
    parser.add_argument("--canny-dir", type=str, default=None)
    parser.add_argument("--mine-label", type=str, default="Mine")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--top-images", type=int, default=10, help="top N images by full-image PSNR gap")
    parser.add_argument("--top-crops", type=int, default=2, help="top N crop regions per image")
    parser.add_argument("--crop-size", type=int, default=100)
    parser.add_argument("--crop-step", type=int, default=50)
    parser.add_argument("--hr-size", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draw-boxes", action="store_true", default=True)
    parser.add_argument("--no-draw-boxes", action="store_false", dest="draw_boxes")
    opt = parser.parse_args()

    if not opt.baseline_dir or not opt.mine_dir:
        raise SystemExit("Need both --baseline-dir and --mine-dir")

    stems = list_stems(opt.mine_dir)
    ranked = rank_images_by_full_psnr_gap(
        stems, opt.gt_dir, opt.baseline_dir, opt.mine_dir, opt.hr_size,
    )
    if not ranked:
        raise SystemExit("No matched triplets (GT, baseline, mine)")

    top_rows = ranked[: opt.top_images]
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mc = MetricsComputer(opt.device)

    summary_lines = [
        "# Auto SR compare: rank by full-image PSNR(Y) gap = mine - baseline",
        f"# crop search: size={opt.crop_size} step={opt.crop_step} top_crops={opt.top_crops}",
        f"{'rank':>4}  {'stem':<8}  {'base':>7}  {'mine':>7}  {'gap':>7}  crops (gap @ x1,y1,x2,y2)",
    ]

    print(f"Scanned {len(ranked)} images; rendering top {len(top_rows)}")
    for rank, row in enumerate(top_rows, 1):
        stem = row["stem"]
        gt = load_rgb(resolve_image(opt.gt_dir, stem), size=opt.hr_size)
        base = load_rgb(resolve_image(opt.baseline_dir, stem), size=opt.hr_size)
        mine = load_rgb(resolve_image(opt.mine_dir, stem), size=opt.hr_size)

        crop_picks = pick_top_boxes_by_gap(
            gt, base, mine,
            crop_size=opt.crop_size,
            crop_step=opt.crop_step,
            top_n=opt.top_crops,
        )
        crops = [(f"Crop {i + 1}", box) for i, (_, box) in enumerate(crop_picks)]

        panels, gt_arr, lq, _ = build_panels(
            stem, opt.lq_dir, opt.gt_dir, opt.canny_dir,
            opt.baseline_dir, opt.mine_dir, opt.hr_size, opt.panel_size,
        )
        panels = [(opt.mine_label if t == "Mine" else t, im) for t, im in panels]

        out_png = out_dir / f"vis_{stem}.png"
        render_comparison(
            stem, panels, gt_arr, lq, crops, opt.mine_label,
            opt.canny_dir, opt.baseline_dir, opt.mine_dir,
            opt.hr_size, opt.panel_size, mc, opt.draw_boxes, out_png,
        )

        crop_str = "  ".join(
            f"{g:+.2f}@{b[0]},{b[1]},{b[2]},{b[3]}" for g, b in crop_picks
        )
        line = (
            f"{rank:4d}  {stem:<8}  {row['psnr_base']:7.3f}  {row['psnr_mine']:7.3f}  "
            f"{row['gap']:+7.3f}  {crop_str}"
        )
        summary_lines.append(line)
        print(line)
        print(f"  -> {out_png}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
