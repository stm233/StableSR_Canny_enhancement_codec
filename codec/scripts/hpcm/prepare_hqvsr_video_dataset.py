#!/usr/bin/env python3
"""Preprocess HQ-VSR videos for two-stage HPCM video codec training.

Stage 1 (I-frame): input/target = R=G=B = binary Canny of the frame.
Stage 2 (P-frame): encoder R=G = prev Canny, B = curr Canny (all Canny, no gray).

Output layout under --out-dir:
  canny/{video_stem}/{frame:06d}.png   # L-mode binary edge map per frame
  manifest_iframe.jsonl
  manifest_pframe.jsonl
  stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def resize_center_crop_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = max(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return rgb[top : top + size, left : left + size]


def compute_canny_l(rgb: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 1.4)
    return cv2.Canny(blur, low, high)


def save_l_png(path: Path, arr_l: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_l, mode="L").save(path)


def process_video(
    video_path: Path,
    out_dir: Path,
    size: int,
    frame_step: int,
    canny_low: int,
    canny_high: int,
    min_frames: int,
    iframe_records: list,
    pframe_records: list,
) -> dict:
    stem = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": stem, "status": "open_failed", "frames": 0}

    frame_idx = 0
    saved_idx = 0
    prev_canny_path = None
    local_iframe = 0
    local_pframe = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        rgb = resize_center_crop_rgb(frame_bgr, size)
        canny_l = compute_canny_l(rgb, canny_low, canny_high)

        canny_rel = f"canny/{stem}/{saved_idx:06d}.png"
        canny_path = out_dir / canny_rel

        if not canny_path.exists():
            save_l_png(canny_path, canny_l)

        iframe_records.append(
            {
                "video": stem,
                "frame": saved_idx,
                "canny": canny_rel,
            }
        )
        local_iframe += 1

        if prev_canny_path is not None:
            pframe_records.append(
                {
                    "video": stem,
                    "frame": saved_idx,
                    "prev_canny": prev_canny_path,
                    "curr_canny": canny_rel,
                }
            )
            local_pframe += 1

        prev_canny_path = canny_rel
        saved_idx += 1
        frame_idx += 1

    cap.release()
    status = "ok" if saved_idx >= min_frames else "too_short"
    return {
        "video": stem,
        "status": status,
        "frames": saved_idx,
        "iframe": local_iframe,
        "pframe": local_pframe,
    }


def write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess HQ-VSR for HPCM video codec.")
    parser.add_argument(
        "--video-root",
        type=str,
        default="/data/Dataset/HQ-VSR",
        help="Directory of source .mp4 files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/data/Dataset/HQ-VSR_processed",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep every N-th frame (1 = consecutive)",
    )
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all videos")
    args = parser.parse_args()

    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p
        for p in video_root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    iframe_records: list = []
    pframe_records: list = []

    print(f"Video root: {video_root}")
    print(f"Output:     {out_dir}")
    print(f"Videos:     {len(videos)}")
    print(f"Size:       {args.size}")
    print(f"Frame step: {args.frame_step}")

    per_video_stats = []
    for video_path in tqdm(videos, desc="videos"):
        stat = process_video(
            video_path=video_path,
            out_dir=out_dir,
            size=args.size,
            frame_step=args.frame_step,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            min_frames=args.min_frames,
            iframe_records=iframe_records,
            pframe_records=pframe_records,
        )
        per_video_stats.append(stat)

    write_jsonl(out_dir / "manifest_iframe.jsonl", iframe_records)
    write_jsonl(out_dir / "manifest_pframe.jsonl", pframe_records)

    ok_videos = sum(1 for s in per_video_stats if s["status"] == "ok")
    stats = {
        "video_root": str(video_root),
        "out_dir": str(out_dir),
        "num_videos": len(videos),
        "ok_videos": ok_videos,
        "iframe_samples": len(iframe_records),
        "pframe_samples": len(pframe_records),
        "size": args.size,
        "frame_step": args.frame_step,
        "per_video": per_video_stats,
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"  I-frame samples: {len(iframe_records)}")
    print(f"  P-frame samples: {len(pframe_records)}")
    print(f"  Stats:           {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
