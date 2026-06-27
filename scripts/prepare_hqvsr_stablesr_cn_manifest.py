#!/usr/bin/env python3
"""Build manifest for StableSR ControlNet training on HQ-VSR_SR_codec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from basicsr.data.hqvsr_paired_lq_canny_dataset import _scan_triplets


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--codec-root", type=Path, default=Path("/data/Dataset/HQ-VSR_SR_codec"))
    p.add_argument("--gt-subdir", default="lossless/hq512")
    p.add_argument("--lq-subdir", default="dcvc_lq_qp0/lq64_lossy")
    p.add_argument("--canny-subdir", default="lossless/canny256")
    args = p.parse_args()

    train = _scan_triplets(args.codec_root, args.gt_subdir, args.lq_subdir, args.canny_subdir)
    write_jsonl(args.codec_root / "manifest_stablesr_cn_train.jsonl", train)

    rng = __import__("random").Random(42)
    val_pool = list(train)
    rng.shuffle(val_pool)
    val = val_pool[:500]
    write_jsonl(args.codec_root / "manifest_stablesr_cn_val.jsonl", val)

    stats = {
        "train_triplets": len(train),
        "val_triplets": len(val),
        "gt_subdir": args.gt_subdir,
        "lq_subdir": args.lq_subdir,
        "canny_subdir": args.canny_subdir,
    }
    out_stats = args.codec_root / "manifest_stablesr_cn_stats.json"
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
