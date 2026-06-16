#!/usr/bin/env python3
"""Create a fixed HQ-VSR test set: 500 I-frames + 500 consecutive P-frame pairs.

Default: 5 videos x 100 frames each (101 consecutive frames per video for P-frame pairs).

Output: /data/Dataset/HQ-VSR_test500/

  hq/000000.png ...
  lq_128/, lq_64/
  canny/          # symlinks to HQ-VSR_processed
  manifest_iframe.jsonl
  manifest_pframe.jsonl
  meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
FRAME_NAME_RE = re.compile(r"^(\d{6})\.png$")


def resize_center_crop_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = max(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return rgb[top : top + size, left : left + size]


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def lq_from_hq(hq_rgb: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(hq_rgb, mode="RGB")
    return np.array(img.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def find_video_path(video_root: Path, stem: str) -> Path:
    for ext in VIDEO_EXTENSIONS:
        p = video_root / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Video not found for stem {stem} under {video_root}")


def load_per_video_stats(processed_root: Path) -> list[dict]:
    stats_path = processed_root / "stats.json"
    with stats_path.open(encoding="utf-8") as f:
        stats = json.load(f)
    return [v for v in stats["per_video"] if v.get("status") == "ok"]


def pick_videos(
    videos: list[dict],
    names: list[str],
    num_videos: int,
    min_frames: int,
) -> list[tuple[str, int]]:
    if names:
        picked = []
        for name in names:
            found = None
            for v in videos:
                if v["video"] == name:
                    found = v
                    break
            if found is None:
                raise FileNotFoundError(f"Video not found in stats: {name}")
            if found["frames"] < min_frames:
                raise ValueError(
                    f"Video {name} has only {found['frames']} frames, need {min_frames}"
                )
            picked.append((found["video"], found["frames"]))
        return picked

    candidates = [(v["video"], v["frames"]) for v in videos if v["frames"] >= min_frames]
    if len(candidates) < num_videos:
        raise RuntimeError(
            f"Need {num_videos} videos with >={min_frames} frames, found {len(candidates)}"
        )
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[:num_videos]


def export_hq_lq_segment(
    video_path: Path,
    out_dir: Path,
    global_base: int,
    start_frame: int,
    num_frames: int,
    hq_size: int,
    lq_sizes: list[int],
    frame_step: int = 1,
) -> list[dict]:
    hq_dir = out_dir / "hq"
    lq_dirs = {s: out_dir / f"lq_{s}" for s in lq_sizes}
    for d in [hq_dir, *lq_dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_infos: list[dict] = []
    frame_idx = 0
    saved_idx = 0
    target_end = start_frame + num_frames - 1

    while saved_idx < num_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if frame_idx > target_end:
            break

        global_i = global_base + saved_idx
        hq_rgb = resize_center_crop_rgb(frame_bgr, hq_size)
        hq_rel = f"hq/{global_i:06d}.png"
        save_rgb_png(out_dir / hq_rel, hq_rgb)

        lq_paths = {}
        for lq_size in lq_sizes:
            lq_rgb = lq_from_hq(hq_rgb, lq_size)
            lq_rel = f"lq_{lq_size}/{global_i:06d}.png"
            save_rgb_png(out_dir / lq_rel, lq_rgb)
            lq_paths[f"lq_{lq_size}"] = lq_rel

        frame_infos.append({
            "global_i": global_i,
            "segment_i": saved_idx,
            "src_frame": frame_idx,
            "hq": hq_rel,
            **lq_paths,
        })
        saved_idx += 1
        frame_idx += 1

    cap.release()
    if saved_idx < num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames from {video_path}, got {saved_idx} "
            f"(start={start_frame}, frame_step={frame_step})"
        )
    return frame_infos


def symlink_canny_segment(
    processed: Path,
    out_dir: Path,
    video: str,
    global_base: int,
    start_frame: int,
    num_frames: int,
) -> list[str]:
    canny_out = out_dir / "canny"
    canny_out.mkdir(parents=True, exist_ok=True)
    src_stem = processed / "canny" / video
    seq_paths: list[str] = []
    for segment_i, src_i in enumerate(range(start_frame, start_frame + num_frames)):
        src = src_stem / f"{src_i:06d}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing source: {src}")
        global_i = global_base + segment_i
        rel = f"canny/{global_i:06d}.png"
        dst = out_dir / rel
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        seq_paths.append(rel)
    return seq_paths


def cleanup_stale_frames(out_dir: Path, keep_max_index: int) -> None:
    for sub in ("hq", "canny", "lq_128", "lq_64"):
        d = out_dir / sub
        if not d.is_dir():
            continue
        for path in d.iterdir():
            if not path.is_file() and not path.is_symlink():
                continue
            m = FRAME_NAME_RE.match(path.name)
            if m and int(m.group(1)) > keep_max_index:
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=str, default="/data/Dataset/HQ-VSR")
    parser.add_argument("--processed-root", type=str, default="/data/Dataset/HQ-VSR_processed")
    parser.add_argument("--out-dir", type=str, default="/data/Dataset/HQ-VSR_test500")
    parser.add_argument(
        "--videos",
        type=str,
        default="",
        help="Comma-separated video stems; default=top-N longest",
    )
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--frames-per-video", type=int, default=100, help="I-frames per video")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame in each video")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--hq-size", type=int, default=512)
    parser.add_argument("--lq-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--skip-hq-lq", action="store_true", help="Only symlink canny manifests")
    args = parser.parse_args()

    processed = Path(args.processed_root)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    frames_per_video = args.frames_per_video
    num_videos = args.num_videos
    segment_frames = frames_per_video + 1  # +1 for last P-frame pair
    total_iframe = num_videos * frames_per_video
    total_pframe = num_videos * frames_per_video
    video_names = [v.strip() for v in args.videos.split(",") if v.strip()]
    if video_names:
        num_videos = len(video_names)

    videos = load_per_video_stats(processed)
    picked = pick_videos(videos, video_names, num_videos, args.start_frame + segment_frames)

    segments: list[dict] = []
    iframe_records: list[dict] = []
    pframe_records: list[dict] = []
    global_base = 0
    global_iframe_seq = 0

    for video_idx, (video, total_frames) in enumerate(picked):
        if args.start_frame + segment_frames > total_frames:
            raise ValueError(
                f"Video {video} has {total_frames} frames, "
                f"need start={args.start_frame} + {segment_frames}"
            )

        frame_infos: list[dict] = []
        if not args.skip_hq_lq:
            video_path = find_video_path(video_root, video)
            print(f"[{video_idx + 1}/{num_videos}] Export HQ/LQ: {video}")
            frame_infos = export_hq_lq_segment(
                video_path=video_path,
                out_dir=out_dir,
                global_base=global_base,
                start_frame=args.start_frame,
                num_frames=segment_frames,
                hq_size=args.hq_size,
                lq_sizes=args.lq_sizes,
                frame_step=args.frame_step,
            )

        canny_paths = symlink_canny_segment(
            processed=processed,
            out_dir=out_dir,
            video=video,
            global_base=global_base,
            start_frame=args.start_frame,
            num_frames=segment_frames,
        )

        for segment_i in range(frames_per_video):
            g = global_base + segment_i
            rec = {
                "video": video,
                "video_idx": video_idx,
                "src_frame": args.start_frame + segment_i,
                "seq": global_iframe_seq,
                "segment_seq": segment_i,
                "canny": canny_paths[segment_i],
            }
            if frame_infos:
                info = frame_infos[segment_i]
                rec["hq"] = info["hq"]
                for lq_size in args.lq_sizes:
                    rec[f"lq_{lq_size}"] = info[f"lq_{lq_size}"]
            iframe_records.append(rec)
            global_iframe_seq += 1

        for segment_i in range(frames_per_video):
            rec = {
                "video": video,
                "video_idx": video_idx,
                "src_frame": args.start_frame + segment_i + 1,
                "seq": video_idx * frames_per_video + segment_i,
                "segment_seq": segment_i + 1,
                "prev_canny": canny_paths[segment_i],
                "curr_canny": canny_paths[segment_i + 1],
            }
            if frame_infos:
                rec["prev_hq"] = frame_infos[segment_i]["hq"]
                rec["curr_hq"] = frame_infos[segment_i + 1]["hq"]
                for lq_size in args.lq_sizes:
                    key = f"lq_{lq_size}"
                    rec[f"prev_{key}"] = frame_infos[segment_i][key]
                    rec[f"curr_{key}"] = frame_infos[segment_i + 1][key]
            pframe_records.append(rec)

        segments.append({
            "video": video,
            "video_idx": video_idx,
            "src_frame_start": args.start_frame,
            "src_frame_end": args.start_frame + segment_frames - 1,
            "global_frame_start": global_base,
            "global_frame_end": global_base + segment_frames - 1,
            "num_iframe": frames_per_video,
            "num_pframe_pairs": frames_per_video,
        })
        global_base += segment_frames

    cleanup_stale_frames(out_dir, keep_max_index=global_base - 1)

    def write_jsonl(path: Path, records: list) -> None:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "manifest_iframe.jsonl", iframe_records)
    write_jsonl(out_dir / "manifest_pframe.jsonl", pframe_records)

    meta = {
        "source_video_root": str(video_root),
        "source_processed_root": str(processed),
        "num_videos": num_videos,
        "frames_per_video": frames_per_video,
        "segment_frames": segment_frames,
        "frame_step": args.frame_step,
        "hq_size": args.hq_size,
        "lq_sizes": args.lq_sizes,
        "num_iframe": total_iframe,
        "num_pframe_pairs": total_pframe,
        "total_frame_files": global_base,
        "segments": segments,
        "note": (
            f"{num_videos} videos x {frames_per_video} I-frames; "
            "P-frame pairs only within each video segment; LQ = bicubic from HQ"
        ),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Created: {out_dir}")
    print(f"  Videos:  {num_videos} x {frames_per_video} I-frames")
    for seg in segments:
        print(f"    - {seg['video']}  frames {seg['src_frame_start']}..{seg['src_frame_end']}")
    print(f"  HQ:      {args.hq_size}x{args.hq_size}")
    print(f"  LQ:      {', '.join(f'{s}x{s}' for s in args.lq_sizes)}")
    print(f"  I-frame: {total_iframe}  P-frame: {total_pframe}")


if __name__ == "__main__":
    main()
