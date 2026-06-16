#!/usr/bin/env python3
"""Plot HPCM direct 512x512 compression R-D curves (6 MSE points)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


LAMBDA_ORDER = ["0.0018", "0.0035", "0.0067", "0.013", "0.025", "0.0483"]
MODEL_LABEL = "HPCM"


def load_bpp(hpcm_root: Path, lam: str) -> float:
    results_json = hpcm_root / f"lambda_{lam}" / "results.json"
    if results_json.is_file():
        with open(results_json) as f:
            return float(json.load(f)["summary"]["bpp"])

    summary_txt = hpcm_root / f"lambda_{lam}" / "summary.txt"
    in_avg = False
    for line in summary_txt.read_text().splitlines():
        line = line.strip()
        if line == "[Average]":
            in_avg = True
            continue
        if in_avg and line.startswith("bpp:"):
            return float(line.split(":", 1)[1].strip())
    raise ValueError(f"bpp not found for lambda={lam}")


def load_eval_metrics(metrics_dir: Path, lam: str) -> dict:
    path = metrics_dir / f"lambda_{lam}.json"
    with open(path) as f:
        return json.load(f)


def collect_points(hpcm_root: Path, metrics_dir: Path):
    rows = []
    for lam in LAMBDA_ORDER:
        m = load_eval_metrics(metrics_dir, lam)
        rows.append({
            "lambda": lam,
            "bpp": load_bpp(hpcm_root, lam),
            "psnr_y": m["psnr_y_mean"],
            "ms_ssim_y": m["ms_ssim_y_mean"],
            "lpips": m["lpips_mean"],
        })
    rows.sort(key=lambda r: r["bpp"])
    return rows


def plot_curve(rows, y_key, ylabel, out_path):
    xs = [r["bpp"] for r in rows]
    ys = [r[y_key] for r in rows]
    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, ys, "-o", linewidth=2, markersize=6, label=MODEL_LABEL)
    plt.xlabel("BPP")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpcm-root", type=str, required=True)
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    hpcm_root = Path(args.hpcm_root)
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_points(hpcm_root, metrics_dir)

    csv_path = out_dir / "hpcm_hr512_rd_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["lambda", "bpp", "psnr_y", "ms_ssim_y", "lpips"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    plot_curve(rows, "psnr_y", "PSNR", out_dir / "psnr_vs_bpp_hpcm512.png")
    plot_curve(rows, "ms_ssim_y", "MS-SSIM", out_dir / "msssim_vs_bpp_hpcm512.png")
    plot_curve(rows, "lpips", "LPIPS", out_dir / "lpips_vs_bpp_hpcm512.png")


if __name__ == "__main__":
    main()
