#!/usr/bin/env python3
"""Build iframe/pframe manifests for HQ-VSR_SR_codec conditional training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def scan_iframe(codec_root: Path, target_subdir: str, cond_rel_prefix: str) -> list[dict]:
    records = []
    root = codec_root / "lossless" / target_subdir
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for png in sorted(clip_dir.glob("*.png")):
            cond = codec_root / cond_rel_prefix / clip_dir.name / png.name
            if not cond.is_file():
                continue
            records.append({
                "video": clip_dir.name,
                "frame": png.stem,
                "target": f"lossless/{target_subdir}/{clip_dir.name}/{png.name}",
                "cond": f"{cond_rel_prefix}/{clip_dir.name}/{png.name}",
            })
    return records


def scan_pframe(codec_root: Path, target_subdir: str, cond_rel_prefix: str) -> list[dict]:
    records = []
    root = codec_root / "lossless" / target_subdir
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted(clip_dir.glob("*.png"))
        prev_rel = None
        for png in frames:
            curr_rel = f"lossless/{target_subdir}/{clip_dir.name}/{png.name}"
            cond_rel = f"{cond_rel_prefix}/{clip_dir.name}/{png.name}"
            if not (codec_root / cond_rel).is_file():
                prev_rel = curr_rel
                continue
            if prev_rel is not None:
                records.append({
                    "video": clip_dir.name,
                    "frame": png.stem,
                    "prev_canny": prev_rel,
                    "curr_canny": curr_rel,
                    "cond": cond_rel,
                })
            prev_rel = curr_rel
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--codec-root", type=Path, default=Path("/data/Dataset/HQ-VSR_SR_codec"))
    p.add_argument(
        "--cond-subdir",
        type=str,
        default="dcvc_lq_qp0/canny64_lossy",
    )
    args = p.parse_args()

    root = args.codec_root
    iframe256 = scan_iframe(root, "canny256", args.cond_subdir)
    iframe128 = scan_iframe(root, "canny128", args.cond_subdir)
    pframe256 = scan_pframe(root, "canny256", args.cond_subdir)
    pframe128 = scan_pframe(root, "canny128", args.cond_subdir)

    write_jsonl(root / "manifest_iframe_canny256.jsonl", iframe256)
    write_jsonl(root / "manifest_iframe_canny128.jsonl", iframe128)
    write_jsonl(root / "manifest_pframe_canny256.jsonl", pframe256)
    write_jsonl(root / "manifest_pframe_canny128.jsonl", pframe128)

    stats = {
        "iframe_canny256": len(iframe256),
        "iframe_canny128": len(iframe128),
        "pframe_canny256": len(pframe256),
        "pframe_canny128": len(pframe128),
    }
    (root / "manifest_codec_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
