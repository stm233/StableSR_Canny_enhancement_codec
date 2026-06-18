#!/usr/bin/env python3
"""Prepare I-frame test manifests from existing frame folders (GT).

This script converts a dataset organized as:

  GT_ROOT/
    clip_001/
      00000.png
      00001.png
      ...
    clip_002/
      ...

into a codec I-frame dataset root:

  OUT_DIR/
    canny/{clip}/{frame:05d or original_name}.png
    manifest_iframe.jsonl
    stats.json

Each manifest record contains:
  {"video": "<clip>", "frame": <index>, "canny": "canny/<clip>/<name>.png"}

The output is compatible with `codec/test_video_iframe.py` which only needs `canny`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NUM_RE = re.compile(r"(\d+)")


def natural_key(p: Path) -> tuple:
    # Sort by first number if present, else lexicographic.
    m = NUM_RE.search(p.stem)
    if m:
        return (int(m.group(1)), p.name)
    return (10**18, p.name)


def resize_center_crop_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = max(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return rgb[top : top + size, left : left + size]


def compute_canny_l(rgb: np.ndarray, low: int, high: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 1.4)
    return cv2.Canny(blur, low, high)


def save_l_png(path: Path, arr_l: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_l, mode="L").save(path)


def iter_clips(gt_root: Path) -> list[Path]:
    clips = [p for p in gt_root.iterdir() if p.is_dir()]
    return sorted(clips, key=lambda p: p.name)


def iter_frames(clip_dir: Path) -> list[Path]:
    frames = [
        p
        for p in clip_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    return sorted(frames, key=natural_key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare I-frame Canny manifests from GT frame folders.")
    ap.add_argument("--gt-root", type=str, required=True, help="Root directory containing clip subfolders.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output dataset root.")
    ap.add_argument("--size", type=int, default=512, help="Resize+center-crop size.")
    ap.add_argument("--frame-step", type=int, default=1, help="Keep every N-th frame.")
    ap.add_argument("--canny-low", type=int, default=100)
    ap.add_argument("--canny-high", type=int, default=200)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all clips")
    ap.add_argument("--max-frames-per-clip", type=int, default=0, help="0 = all frames")
    ap.add_argument("--overwrite", action="store_true", help="Recompute canny even if exists.")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = iter_clips(gt_root)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    records: list[dict] = []
    per_clip = []
    total_frames = 0

    for clip_dir in tqdm(clips, desc="clips"):
        frames = iter_frames(clip_dir)
        if args.max_frames_per_clip > 0:
            frames = frames[: args.max_frames_per_clip]

        kept = 0
        for idx, frame_path in enumerate(frames):
            if idx % args.frame_step != 0:
                continue

            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            rgb = resize_center_crop_rgb(frame_bgr, args.size)
            canny_l = compute_canny_l(rgb, args.canny_low, args.canny_high)

            # Keep original frame filename (safer across datasets: 00000.png, 00001.png, ...).
            rel = f"canny/{clip_dir.name}/{frame_path.name}"
            out_path = out_dir / rel
            if args.overwrite or not out_path.exists():
                save_l_png(out_path, canny_l)

            records.append({"video": clip_dir.name, "frame": idx, "canny": rel})
            kept += 1

        total_frames += kept
        per_clip.append({"clip": clip_dir.name, "frames": len(frames), "kept": kept})

    (out_dir / "manifest_iframe.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )

    stats = {
        "gt_root": str(gt_root),
        "out_dir": str(out_dir),
        "size": args.size,
        "frame_step": args.frame_step,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "num_clips": len(clips),
        "iframe_samples": len(records),
        "kept_frames": total_frames,
        "per_clip": per_clip,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"  Out:     {out_dir}")
    print(f"  Clips:   {len(clips)}")
    print(f"  Samples: {len(records)} (all I-frames)")
    print(f"  Manifest:{out_dir / 'manifest_iframe.jsonl'}")


if __name__ == "__main__":
    main()

