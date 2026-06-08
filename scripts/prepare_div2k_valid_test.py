#!/usr/bin/env python3
"""
Prepare DIV2K validation set for 128->512 SR test (100 images).

DIV2K does NOT ship official blind-SR LQ. For synthetic benchmark (paper DIV2K Valid),
LQ is bicubic downscale of HR (128x128); HQ is 512x512 center crop/resize.

Outputs under /data/Dataset/DIV2K/DIV2K_valid_100_512_128/:
  HR_512/   GT 512x512
  LR_128/   LQ 128x128 (bicubic from HR_512)
  canny/    binary edges from HR (OpenCV, same as canny.py)

Usage:
  python scripts/prepare_div2k_valid_test.py
  python scripts/prepare_div2k_valid_test.py --num 100 --hq-size 512 --lq-size 128
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ldm.canny_util import compute_binary_canny_bgr

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = max(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/Dataset/DIV2K")
    parser.add_argument("--valid-hr", type=str, default=None, help="DIV2K_valid_HR folder")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--hq-size", type=int, default=512)
    parser.add_argument("--lq-size", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.root)
    valid_hr = Path(args.valid_hr) if args.valid_hr else root / "DIV2K_valid_HR"
    out_root = Path(args.out) if args.out else root / "DIV2K_valid_100_512_128"
    hr_dir = out_root / "HR_512"
    lq_dir = out_root / f"LR_{args.lq_size}"
    canny_dir = out_root / "canny"
    for d in (hr_dir, lq_dir, canny_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not valid_hr.is_dir():
        print(f"Missing {valid_hr}. Run: python scripts/prepare_training_datasets.py")
        sys.exit(1)

    paths = sorted(
        p for p in valid_hr.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[: args.num]
    print(f"Processing {len(paths)} images -> {out_root}")

    for i, src in enumerate(paths, 1):
        name = f"{src.stem}.png"
        print(f"[{i}/{len(paths)}] {src.name}")
        img = Image.open(src).convert("RGB")
        hq = resize_center_crop(img, args.hq_size)
        lq = hq.resize((args.lq_size, args.lq_size), Image.BICUBIC)
        hq.save(hr_dir / name)
        lq.save(lq_dir / name)
        bgr = cv2.cvtColor(np.array(hq), cv2.COLOR_RGB2BGR)
        edge = compute_binary_canny_bgr(bgr)
        cv2.imwrite(str(canny_dir / name), edge)

    print(f"Saved HR {args.hq_size}x{args.hq_size} -> {hr_dir}")
    print(f"Saved LR {args.lq_size}x{args.lq_size} -> {lq_dir}")
    print(f"Saved binary canny -> {canny_dir}")


if __name__ == "__main__":
    main()
