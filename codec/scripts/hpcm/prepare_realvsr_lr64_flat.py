#!/usr/bin/env python3
"""Build flat LR64 + GT512 folders for RealVSR I-frame codec / StableSR pairing.

Reads manifest_iframe.jsonl (video/frame/canny), loads GT RGB frames, center-crops
to 512, bicubic-downsamples to 64 for HPCM_Base LQ codec. Filenames:
  {video}_{frame_stem}.png  e.g. 016_00000.png

Outputs:
  OUT_DIR/lr64/   64x64 RGB  -> HPCM_Base --init-img
  OUT_DIR/gt512/  512x512 RGB -> StableSR --gt-img
  OUT_DIR/manifest_flat.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_center_crop_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = max(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return rgb[top : top + size, left : left + size]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True, help="GT_test/<clip>/xxxx.png")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--lr-size", type=int, default=64)
    p.add_argument("--gt-size", type=int, default=512)
    args = p.parse_args()

    records = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    lr_dir = args.out_dir / "lr64"
    gt_dir = args.out_dir / "gt512"
    lr_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    flat_manifest = []
    for rec in tqdm(records, desc="prepare lr64/gt512"):
        video = rec["video"]
        frame_stem = Path(rec["canny"]).stem
        fname = f"{video}_{frame_stem}.png"
        gt_path = args.gt_root / video / f"{frame_stem}.png"
        if not gt_path.exists():
            for ext in (".png", ".jpg", ".jpeg"):
                alt = args.gt_root / video / f"{frame_stem}{ext}"
                if alt.exists():
                    gt_path = alt
                    break
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)

        bgr = cv2.imread(str(gt_path))
        gt512 = resize_center_crop_rgb(bgr, args.gt_size)
        lr64 = cv2.resize(gt512, (args.lr_size, args.lr_size), interpolation=cv2.INTER_CUBIC)

        Image.fromarray(gt512).save(gt_dir / fname)
        Image.fromarray(lr64).save(lr_dir / fname)
        flat_manifest.append({
            "video": video,
            "frame": rec["frame"],
            "name": fname,
            "lr64": f"lr64/{fname}",
            "gt512": f"gt512/{fname}",
            "canny": rec["canny"],
        })

    out_manifest = args.out_dir / "manifest_flat.jsonl"
    with out_manifest.open("w", encoding="utf-8") as f:
        for row in flat_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(flat_manifest)} pairs -> {args.out_dir}")
    print(f"  lr64:  {lr_dir}")
    print(f"  gt512: {gt_dir}")
    print(f"  manifest: {out_manifest}")


if __name__ == "__main__":
    main()
