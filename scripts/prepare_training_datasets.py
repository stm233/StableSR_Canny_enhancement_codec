#!/usr/bin/env python3
"""
Download / organize StableSR training data under /data/Dataset/ (paper + official yaml).

Training GT sources (RealESRGAN on-the-fly degradation; only HR folders needed):
  - DIV2K_train_HR
  - df2k_ost/GT  (= DIV2K_train + Flickr2K + OST images)
  - DIV8K/train_HR
  - FFHQ 1024 (optional 10000 faces)

Usage:
  python scripts/prepare_training_datasets.py --root /data/Dataset
  python scripts/prepare_training_datasets.py --root /data/Dataset --skip-download  # only merge df2k_ost
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("/data/Dataset")

URLS = {
    "div2k_train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "div2k_valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "flickr2k": "https://data.vision.ee.ethz.ch/cvl/DIV2K/Flickr2K_HR.zip",
    "ost": "https://github.com/xinntao/OST_dataset/releases/download/v0.1.0/OST_dataset.zip",
    "div8k_train": "https://github.com/JeffWang987/Open-World-Vision/releases/download/v0.1.0/DIV8K_train_HR.zip",
}


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def download_and_unzip(url, dest_dir, zip_name=None):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zip_name is None:
        zip_name = url.split("/")[-1]
    zip_path = dest_dir / zip_name
    if not zip_path.is_file():
        run(["wget", "-c", url, "-O", str(zip_path)])
    run(["unzip", "-n", str(zip_path), "-d", str(dest_dir)])


def merge_df2k_ost(root: Path):
    """Build df2k_ost/GT like Real-ESRGAN / StableSR official config."""
    out = root / "df2k_ost" / "GT"
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        root / "DIV2K" / "DIV2K_train_HR",
        root / "Flickr2K" / "Flickr2K_HR",
        root / "OST" / "images",
        root / "OST_dataset" / "images",
    ]
    n = 0
    for src in sources:
        if not src.is_dir():
            print(f"  skip missing: {src}")
            continue
        for p in sorted(src.glob("*.png")) + sorted(src.glob("*.jpg")):
            link = out / p.name
            if link.exists():
                continue
            try:
                os.link(p, link)
            except OSError:
                shutil.copy2(p, link)
            n += 1
    print(f"df2k_ost/GT: {len(list(out.glob('*')))} files ({n} new links/copies)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--ffhq", action="store_true", help="Also note FFHQ download (manual, large)")
    args = parser.parse_args()
    root = Path(args.root)

    if not args.skip_download:
        download_and_unzip(URLS["div2k_train"], root / "DIV2K")
        download_and_unzip(URLS["div2k_valid"], root / "DIV2K")
        download_and_unzip(URLS["flickr2k"], root / "Flickr2K")
        try:
            download_and_unzip(URLS["ost"], root)
        except subprocess.CalledProcessError:
            print("OST download failed — get OST from Real-ESRGAN docs and place under OST/images")
        try:
            download_and_unzip(URLS["div8k_train"], root / "DIV8K")
        except subprocess.CalledProcessError:
            print("DIV8K download failed — place train_HR under DIV8K/train_HR manually")

    merge_df2k_ost(root)

    if args.ffhq:
        print(
            "FFHQ: download 1024x1024 from https://github.com/NVlabs/ffhq-dataset\n"
            f"  then extract subset to {root / 'FFHQ' / '1024'}"
        )

    print("\nDone. Training gt_path example:")
    print(f"  - {root / 'DIV8K' / 'train_HR'}")
    print(f"  - {root / 'df2k_ost' / 'GT'}")
    print(f"  - face_gt_path: {root / 'FFHQ' / '1024'}")


if __name__ == "__main__":
    main()
