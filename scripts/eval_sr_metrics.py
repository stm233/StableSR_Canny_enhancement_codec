"""Evaluate SR outputs: PSNR/SSIM/MS-SSIM (Y & RGB), LPIPS, FID."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3
from scripts.util_image import calculate_ms_ssim, calculate_psnr, calculate_ssim
from taming.modules.losses.lpips import LPIPS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder):
    names = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f.lower())[1]
        if ext in IMAGE_EXTENSIONS and not f.endswith("_canny.png"):
            names.append(f)
    return names


def resolve_gt_path(gt_dir, fname):
    gt_path = os.path.join(gt_dir, fname)
    if os.path.exists(gt_path):
        return gt_path
    base = os.path.splitext(fname)[0]
    for ext in IMAGE_EXTENSIONS:
        alt = os.path.join(gt_dir, base + ext)
        if os.path.exists(alt):
            return alt
    return None


def resolve_out_path(out_dir, fname):
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path):
        return out_path
    base = os.path.splitext(fname)[0]
    alt = os.path.join(out_dir, base + ".png")
    if os.path.exists(alt):
        return alt
    return None


def load_rgb_float(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32)


def load_rgb_tensor_minus1_1(path, device):
    arr = load_rgb_float(path) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    t = t * 2.0 - 1.0
    return t.to(device)


class InceptionImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = load_rgb_float(self.paths[idx]) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        return t


def compute_activation_stats(features):
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def main():
    parser = argparse.ArgumentParser(description="Evaluate SR folder against GT")
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--json-out", type=str, default="", help="Save metrics summary to JSON")
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    pairs = []
    for fname in list_images(opt.out_dir):
        gt_path = resolve_gt_path(opt.gt_dir, fname)
        out_path = resolve_out_path(opt.out_dir, fname)
        if gt_path is None or out_path is None:
            continue
        pairs.append((fname, gt_path, out_path))

    if not pairs:
        print("No matched GT/output pairs found.")
        return

    print(f"Evaluating {len(pairs)} pairs")
    print(f"  gt:  {opt.gt_dir}")
    print(f"  out: {opt.out_dir}")

    psnrs_y, ssims_y, mssims_y = [], [], []
    psnrs_rgb, ssims_rgb, mssims_rgb = [], [], []
    lpips_scores = []

    lpips_model = None
    if not opt.skip_lpips:
        lpips_model = LPIPS().eval().to(device)

    for _, gt_path, out_path in tqdm(pairs, desc="PSNR/SSIM/MS-SSIM/LPIPS"):
        gt = load_rgb_float(gt_path)
        out = load_rgb_float(out_path)
        if gt.shape != out.shape:
            out_img = Image.fromarray(out.astype(np.uint8))
            out_img = out_img.resize((gt.shape[1], gt.shape[0]), resample=Image.LANCZOS)
            out = np.array(out_img).astype(np.float32)

        psnrs_y.append(calculate_psnr(out, gt, border=0, ycbcr=True))
        ssims_y.append(calculate_ssim(out, gt, border=0, ycbcr=True))
        mssims_y.append(calculate_ms_ssim(out, gt, border=0, ycbcr=True))
        psnrs_rgb.append(calculate_psnr(out, gt, border=0, ycbcr=False))
        ssims_rgb.append(calculate_ssim(out, gt, border=0, ycbcr=False))
        mssims_rgb.append(calculate_ms_ssim(out, gt, border=0, ycbcr=False))

        if lpips_model is not None:
            gt_t = load_rgb_tensor_minus1_1(gt_path, device)
            out_t = load_rgb_tensor_minus1_1(out_path, device)
            if gt_t.shape[-2:] != out_t.shape[-2:]:
                out_t = torch.nn.functional.interpolate(
                    out_t, size=gt_t.shape[-2:], mode="bicubic", align_corners=False
                )
            with torch.no_grad():
                score = lpips_model(out_t, gt_t).mean().item()
            lpips_scores.append(score)

    print(f"matched: {len(pairs)}")
    print(f"PSNR(Y) mean:     {float(np.mean(psnrs_y)):.4f}")
    print(f"SSIM(Y) mean:     {float(np.mean(ssims_y)):.4f}")
    print(f"MS-SSIM(Y) mean:  {float(np.mean(mssims_y)):.4f}")
    print(f"PSNR(RGB) mean:   {float(np.mean(psnrs_rgb)):.4f}")
    print(f"SSIM(RGB) mean:   {float(np.mean(ssims_rgb)):.4f}")
    print(f"MS-SSIM(RGB) mean:{float(np.mean(mssims_rgb)):.4f}")
    if lpips_scores:
        print(f"LPIPS mean:     {float(np.mean(lpips_scores)):.4f}  (lower is better)")

    summary = {
        "matched": len(pairs),
        "psnr_y_mean": float(np.mean(psnrs_y)),
        "ssim_y_mean": float(np.mean(ssims_y)),
        "ms_ssim_y_mean": float(np.mean(mssims_y)),
        "psnr_rgb_mean": float(np.mean(psnrs_rgb)),
        "ssim_rgb_mean": float(np.mean(ssims_rgb)),
        "ms_ssim_rgb_mean": float(np.mean(mssims_rgb)),
        "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
        "gt_dir": opt.gt_dir,
        "out_dir": opt.out_dir,
    }
    if opt.json_out:
        import json
        out_path = os.path.abspath(opt.json_out)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved metrics JSON: {out_path}")

    if opt.skip_fid:
        return

    gt_paths = [p[1] for p in pairs]
    out_paths = [p[2] for p in pairs]
    inception = load_patched_inception_v3(device=device, resize_input=True, normalize_input=True)

    gt_loader = DataLoader(
        InceptionImageDataset(gt_paths),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
    )
    out_loader = DataLoader(
        InceptionImageDataset(out_paths),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
    )

    gt_features = extract_inception_features(
        gt_loader, inception, len_generator=len(gt_loader), device=device
    )
    out_features = extract_inception_features(
        out_loader, inception, len_generator=len(out_loader), device=device
    )
    mu_gt, sigma_gt = compute_activation_stats(gt_features)
    mu_out, sigma_out = compute_activation_stats(out_features)
    fid = calculate_fid(mu_gt, sigma_gt, mu_out, sigma_out)
    print(f"FID:            {float(fid):.4f}  (lower is better)")


if __name__ == "__main__":
    main()
