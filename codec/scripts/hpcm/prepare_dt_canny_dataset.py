#!/usr/bin/env python3
"""Precompute DT RGB PNGs from binary Canny (optional cache for faster IO)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

CODEC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODEC_ROOT))

from src.utils.distance_transform import canny_to_dt_rgb  # noqa: E402


def load_edge01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def save_dt_rgb(path: Path, rgb: np.ndarray) -> None:
    """rgb: [3,H,W] float [0,1] -> uint8 PNG."""
    u8 = (np.clip(rgb, 0, 1) * 255.0).round().astype(np.uint8)
    img = np.transpose(u8, (1, 2, 0))
    Image.fromarray(img, mode="RGB").save(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Canny L PNG -> DT RGB PNG cache.")
    p.add_argument("--canny-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--ext", type=str, default=".png")
    args = p.parse_args()

    src = Path(args.canny_dir)
    dst = Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    print(f"{len(files)} images: {src} -> {dst}")

    for f in tqdm(files):
        edge = load_edge01(f)
        rgb = canny_to_dt_rgb(edge).numpy()
        save_dt_rgb(dst / f"{f.stem}{args.ext}", rgb)


if __name__ == "__main__":
    main()
