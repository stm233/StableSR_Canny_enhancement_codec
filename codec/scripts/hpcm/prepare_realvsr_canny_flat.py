#!/usr/bin/env python3
"""Flatten RealVSR per-clip canny to StableSR naming: {clip}_{frame:05d}.png"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canny-root", type=Path,
                   default=Path("/data/Dataset/RealVSR_GT_test_iframe_all/canny"))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for clip_dir in sorted(p for p in args.canny_root.iterdir() if p.is_dir()):
        clip = clip_dir.name
        for src in sorted(clip_dir.glob("*.png")):
            frame_stem = src.stem
            dst = args.out_dir / f"{clip}_{frame_stem}.png"
            shutil.copy2(src, dst)
            count += 1
    print(f"Wrote {count} canny PNGs -> {args.out_dir}")


if __name__ == "__main__":
    main()
