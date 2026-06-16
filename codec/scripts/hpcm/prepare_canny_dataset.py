#!/usr/bin/env python3
"""Generate 512x512 binary Canny maps (3-channel PNG) for HPCM canny codec training."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = max(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def compute_binary_canny_rgb(rgb: np.ndarray, low=100, high=200) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 1.4)
    edge = cv2.Canny(blur, low, high)
    edge3 = np.stack([edge, edge, edge], axis=2)
    return edge3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hr-root",
        type=str,
        default="/data/Dataset/df2k_ost/GT",
        help="Folder of HR RGB images",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/data/Dataset/HPCM_canny_train",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    hr_root = Path(args.hr_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in hr_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.max_images > 0:
        paths = paths[: args.max_images]

    print(f"HR root: {hr_root}")
    print(f"Output:  {out_dir}")
    print(f"Images:  {len(paths)}")

    for src in tqdm(paths, desc="canny"):
        dst = out_dir / f"{src.stem}.png"
        if dst.exists():
            continue
        img = Image.open(src).convert("RGB")
        hq = resize_center_crop(img, args.size)
        rgb = np.array(hq)
        edge3 = compute_binary_canny_rgb(rgb)
        Image.fromarray(edge3).save(dst)

    print(f"Done. Saved under {out_dir}")


if __name__ == "__main__":
    main()
