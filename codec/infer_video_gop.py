#!/usr/bin/env python3
"""GOP video inference: I + N×P with HPCM_DT1ch / HPCM_Video_PFrame_DT1ch (bitstream path)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.distance_transform import canny_to_dt_rgb  # noqa: E402
from utils import psnr_continuous  # noqa: E402
from test import (  # noqa: E402
    AverageMeter,
    _save_results,
    _sync,
    crop,
    get_scale_table,
    pad,
)


def load_canny_1ch(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    return ToTensor()(img).unsqueeze(0)


def load_dt_rgb(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dt_rgb [1,3,H,W], gt_dist [1,1,H,W])."""
    edge = load_canny_1ch(path)
    dt = canny_to_dt_rgb(edge.squeeze(0)).unsqueeze(0)
    return dt, dt[:, 0:1]


def discover_videos(canny_root: Path) -> dict[str, list[Path]]:
    videos: dict[str, list[Path]] = {}
    if not canny_root.is_dir():
        return videos
    for vdir in sorted(canny_root.iterdir()):
        if not vdir.is_dir():
            continue
        frames = sorted(
            p for p in vdir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if frames:
            videos[vdir.name] = frames
    return videos


def bitstream_bpp(strings, h: int, w: int) -> float:
    return sum(len(s) for s in strings) * 8.0 / (h * w)


@dataclass
class FrameResult:
    video: str
    frame_idx: int
    frame_type: str
    psnr: float
    bpp: float
    y_bpp: float
    z_bpp: float
    enc_time: float
    dec_time: float


@dataclass
class GopResult:
    video: str
    gop_idx: int
    num_p: int
    frames: list[FrameResult] = field(default_factory=list)

    @property
    def avg_psnr(self) -> float:
        return sum(f.psnr for f in self.frames) / max(len(self.frames), 1)

    @property
    def avg_bpp(self) -> float:
        return sum(f.bpp for f in self.frames) / max(len(self.frames), 1)


def parse_args():
    p = argparse.ArgumentParser(description="GOP infer: I + N P-frames (DT1ch bitstream).")
    p.add_argument(
        "--pframe-checkpoint",
        type=str,
        required=True,
        help="HPCM_Video_PFrame_DT1ch checkpoint (includes iframe_codec + codec)",
    )
    p.add_argument(
        "--iframe-checkpoint",
        type=str,
        default="",
        help="Optional separate I-frame ckpt; defaults to weights inside pframe checkpoint",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default="/data/Dataset/HQ-VSR_processed",
    )
    p.add_argument(
        "--num-p",
        type=int,
        default=7,
        help="Number of P-frames after each I-frame (GOP = 1 + num_p). E.g. 4 -> IPPPP",
    )
    p.add_argument("--max-videos", type=int, default=0, help="0 = all videos")
    p.add_argument("--max-gops-per-video", type=int, default=0, help="0 = all GOPs")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num", type=int, default=60, help="entropy scale table levels")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--outdir", type=str, default="", help="Save recon PNGs")
    return p.parse_args()


@torch.no_grad()
def encode_decode_i(model, x_dt: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, dict, FrameResult, tuple[int, int]]:
    h, w = x_dt.size(2), x_dt.size(3)
    x_pad = pad(x_dt)

    _sync(device)
    t0 = time.time()
    enc = model.compress_i(x_pad)
    _sync(device)
    enc_t = time.time() - t0

    _sync(device)
    t0 = time.time()
    dec = model.decompress_i(enc["strings"], enc["shape"])
    _sync(device)
    dec_t = time.time() - t0

    x_hat = crop(dec["x_hat"], (h, w))
    gt = x_dt[:, 0:1]
    psnr = psnr_continuous(x_hat, gt, peak=255.0).item()
    bpp = bitstream_bpp(enc["strings"], h, w)
    fr = FrameResult(
        video="",
        frame_idx=0,
        frame_type="I",
        psnr=psnr,
        bpp=bpp,
        y_bpp=len(enc["strings"][0]) * 8.0 / (h * w),
        z_bpp=len(enc["strings"][1]) * 8.0 / (h * w),
        enc_time=enc_t,
        dec_time=dec_t,
    )
    return x_hat, dec["ref_feats"], fr, (h, w)


@torch.no_grad()
def encode_decode_p(
    model,
    prev_dist: torch.Tensor,
    curr_canny_path: Path,
    ref_feats: dict,
    device: torch.device,
    frame_idx: int,
) -> tuple[torch.Tensor, dict, FrameResult]:
    curr_edge = load_canny_1ch(curr_canny_path).to(device)
    _, gt_dist = load_dt_rgb(curr_canny_path)
    gt_dist = gt_dist.to(device)

    p_in = model.build_p_input_infer(prev_dist, curr_edge)
    h, w = p_in.size(2), p_in.size(3)
    p_pad = pad(p_in)

    _sync(device)
    t0 = time.time()
    enc = model.compress_p(p_pad, ref_feats)
    _sync(device)
    enc_t = time.time() - t0

    _sync(device)
    t0 = time.time()
    dec = model.decompress_p(enc["strings"], enc["shape"], ref_feats)
    _sync(device)
    dec_t = time.time() - t0

    x_hat = crop(dec["x_hat"], (h, w))
    psnr = psnr_continuous(x_hat, gt_dist, peak=255.0).item()
    bpp = bitstream_bpp(enc["strings"], h, w)
    fr = FrameResult(
        video="",
        frame_idx=frame_idx,
        frame_type="P",
        psnr=psnr,
        bpp=bpp,
        y_bpp=len(enc["strings"][0]) * 8.0 / (h * w),
        z_bpp=len(enc["strings"][1]) * 8.0 / (h * w),
        enc_time=enc_t,
        dec_time=dec_t,
    )
    return x_hat, dec["ref_feats"], fr


def run_gop(
    model,
    video: str,
    frames: list[Path],
    gop_idx: int,
    num_p: int,
    device: torch.device,
    outdir: Path | None,
) -> GopResult | None:
    gop_len = 1 + num_p
    start = gop_idx * gop_len
    if start >= len(frames):
        return None
    chunk = frames[start : start + gop_len]
    if len(chunk) < 1:
        return None

    gop = GopResult(video=video, gop_idx=gop_idx, num_p=num_p)

    x_dt, _ = load_dt_rgb(chunk[0])
    x_dt = x_dt.to(device)
    prev_dist, ref_feats, fr_i, _ = encode_decode_i(model, x_dt, device)
    fr_i.video = video
    fr_i.frame_idx = start
    gop.frames.append(fr_i)

    if outdir is not None:
        from test_video_iframe import tensor_to_image
        tensor_to_image(prev_dist).save(outdir / f"{video}_f{start:06d}_I.png")

    for pi, p_path in enumerate(chunk[1:], start=1):
        prev_dist, ref_feats, fr_p = encode_decode_p(
            model, prev_dist, p_path, ref_feats, device, start + pi
        )
        fr_p.video = video
        gop.frames.append(fr_p)
        if outdir is not None:
            from test_video_iframe import tensor_to_image
            tensor_to_image(prev_dist).save(outdir / f"{video}_f{start + pi:06d}_P.png")

    return gop


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(args.device)

    root = Path(args.dataset_root)
    videos = discover_videos(root / "canny")
    if not videos:
        raise RuntimeError(f"No videos under {root / 'canny'}")

    video_names = sorted(videos)
    if args.max_videos > 0:
        video_names = video_names[: args.max_videos]

    results_dir = args.results_dir or args.outdir
    outdir = Path(args.outdir) if args.outdir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    net = importlib.import_module(".HPCM_Video_PFrame_DT1ch", "src.models").HPCM
    model = net(use_lossy_ref=True).eval()
    ckpt = torch.load(args.pframe_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    if args.iframe_checkpoint:
        model.load_iframe_checkpoint(args.iframe_checkpoint, map_location="cpu")
    scale_table = get_scale_table(0.12, 64, args.num)
    model.codec.update(scale_table)
    model.iframe_codec.update(scale_table)
    for p in model.iframe_codec.parameters():
        p.requires_grad = False
    model.iframe_codec.eval()
    model = model.to(device)

    print(
        f"GOP pattern: I + {args.num_p}P  (GOP size = {1 + args.num_p})\n"
        f"Videos: {len(video_names)}  device: {device}\n"
        f"P-frame ckpt: {args.pframe_checkpoint}"
    )

    all_frames: list[FrameResult] = []
    all_gops: list[GopResult] = []
    psnr_i = AverageMeter()
    psnr_p = AverageMeter()
    bpp_i = AverageMeter()
    bpp_p = AverageMeter()
    gop_psnr = AverageMeter()
    gop_bpp = AverageMeter()

    for vname in video_names:
        frames = videos[vname]
        max_gops = (len(frames) + args.num_p) // (1 + args.num_p)
        if args.max_gops_per_video > 0:
            max_gops = min(max_gops, args.max_gops_per_video)

        for gi in range(max_gops):
            gop = run_gop(model, vname, frames, gi, args.num_p, device, outdir)
            if gop is None or not gop.frames:
                continue
            all_gops.append(gop)
            gop_psnr.update(gop.avg_psnr)
            gop_bpp.update(gop.avg_bpp)
            for fr in gop.frames:
                all_frames.append(fr)
                if fr.frame_type == "I":
                    psnr_i.update(fr.psnr)
                    bpp_i.update(fr.bpp)
                else:
                    psnr_p.update(fr.psnr)
                    bpp_p.update(fr.bpp)
            print(
                f"{vname} GOP#{gi}: {len(gop.frames)} frames  "
                f"PSNR={gop.avg_psnr:.2f}  bpp={gop.avg_bpp:.4f}"
            )

    summary = {
        "num_p": args.num_p,
        "gop_size": 1 + args.num_p,
        "videos": len(video_names),
        "gops": len(all_gops),
        "frames": len(all_frames),
        "psnr_i": float(psnr_i.avg),
        "psnr_p": float(psnr_p.avg),
        "bpp_i": float(bpp_i.avg),
        "bpp_p": float(bpp_p.avg),
        "gop_psnr_avg": float(gop_psnr.avg),
        "gop_bpp_avg": float(gop_bpp.avg),
    }
    print(
        f"\nGOP Infer ({summary['gops']} GOPs, {summary['frames']} frames):"
        f"\n  I-frame PSNR: {summary['psnr_i']:.4f}  bpp: {summary['bpp_i']:.6f}"
        f"\n  P-frame PSNR: {summary['psnr_p']:.4f}  bpp: {summary['bpp_p']:.6f}"
        f"\n  GOP avg PSNR: {summary['gop_psnr_avg']:.4f}  bpp: {summary['gop_bpp_avg']:.6f}"
    )

    if results_dir:
        per_frame = [
            {
                "video": f.video,
                "frame": f.frame_idx,
                "type": f.frame_type,
                "psnr": f.psnr,
                "bpp": f.bpp,
                "y_bpp": f.y_bpp,
                "z_bpp": f.z_bpp,
                "enc_time": f.enc_time,
                "dec_time": f.dec_time,
            }
            for f in all_frames
        ]
        out_path = Path(results_dir) / f"gop_infer_numP{args.num_p}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "frames": per_frame}, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
