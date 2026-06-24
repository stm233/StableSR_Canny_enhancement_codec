#!/usr/bin/env python3
"""Precompute lossy I-frame refs for P-frame DT1ch training (per unique prev frame)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

CODEC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODEC_ROOT))

from src.datasets.video_codec_dataset import (  # noqa: E402
    _load_manifest,
    _prev_canny_rel,
    pframe_ref_cache_path,
)
from src.models import HPCM_DT1ch  # noqa: E402
from src.models.codec_fusion import lossy_iframe_ref_bundle  # noqa: E402
from src.utils.distance_transform import canny_to_dt_rgb  # noqa: E402
from utils import get_scale_table  # noqa: E402


def g_from_r_hat(r_hat: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    out = []
    for b in range(r_hat.size(0)):
        edge = (r_hat[b, 0] >= threshold).float()
        dt = canny_to_dt_rgb(edge)
        out.append(dt[1:2])
    return torch.stack(out, dim=0)


def load_canny_1ch(path: Path) -> torch.Tensor:
    return ToTensor()(Image.open(path).convert("L")).unsqueeze(0)


def ckpt_tag(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def parse_args():
    p = argparse.ArgumentParser(description="Build P-frame I-frame ref cache.")
    p.add_argument("--iframe-checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--manifest", type=str, default="manifest_pframe.jsonl")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num", type=int, default=60, help="entropy scale table levels")
    p.add_argument("--edge-threshold", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all unique prev frames")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    data_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    records = _load_manifest(data_root / args.manifest)
    prev_paths = sorted({_prev_canny_rel(r) for r in records})
    if args.max_frames > 0:
        prev_paths = prev_paths[: args.max_frames]

    model = HPCM_DT1ch.HPCM().eval()
    ckpt = torch.load(args.iframe_checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.update(get_scale_table(0.12, 64, args.num))
    model = model.to(device)

    meta = {
        "iframe_checkpoint": str(Path(args.iframe_checkpoint).resolve()),
        "iframe_ckpt_tag": ckpt_tag(args.iframe_checkpoint),
        "dataset_root": str(data_root.resolve()),
        "manifest": args.manifest,
        "edge_threshold": args.edge_threshold,
        "num_prev_frames": len(prev_paths),
        "encoding": "inverted_r_edge_1",
    }
    (out_dir / "cache_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    index = []
    skipped = 0
    for prev_rel in tqdm(prev_paths, desc="prev frames"):
        cache_path = pframe_ref_cache_path(out_dir, prev_rel)
        if cache_path.is_file() and not args.overwrite:
            skipped += 1
            index.append({"prev_canny": prev_rel, "cache": str(cache_path.relative_to(out_dir))})
            continue

        canny_path = data_root / prev_rel
        if not canny_path.is_file():
            raise FileNotFoundError(f"Missing prev canny: {canny_path}")

        edge = load_canny_1ch(canny_path).to(device)
        ref_dt = canny_to_dt_rgb(edge.squeeze(0)).unsqueeze(0).to(device)

        prev_r_hat, ref_feats = lossy_iframe_ref_bundle(model, ref_dt)
        prev_g_hat = g_from_r_hat(prev_r_hat, threshold=args.edge_threshold)

        payload = {
            "prev_canny": prev_rel,
            "prev_r_hat": prev_r_hat.squeeze(0).cpu(),
            "prev_g_hat": prev_g_hat.squeeze(0).cpu(),
            "ref_feats": {k: v.squeeze(0).cpu() for k, v in ref_feats.items()},
            "edge_threshold": args.edge_threshold,
        }
        torch.save(payload, cache_path)
        index.append({"prev_canny": prev_rel, "cache": str(cache_path.relative_to(out_dir))})

    (out_dir / "cache_index.jsonl").write_text(
        "\n".join(json.dumps(row) for row in index) + ("\n" if index else ""),
        encoding="utf-8",
    )

    print(f"Cache root:     {out_dir}")
    print(f"Unique prev:    {len(prev_paths)}")
    print(f"Skipped exists: {skipped}")
    print(f"Written:        {len(prev_paths) - skipped}")
    print(f"Meta:           {out_dir / 'cache_meta.json'}")


if __name__ == "__main__":
    main()
