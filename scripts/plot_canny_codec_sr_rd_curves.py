#!/usr/bin/env python3
"""Plot SR R-D curves: uncompressed LR + HPCM-compressed canny hints (no HPCM@512 baseline)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MSE_LAMBDAS = ["0.0018", "0.0035", "0.0067", "0.013", "0.025", "0.0483"]
MSSSIM_LAMBDAS = ["2.4", "4.58", "8.73", "16.64", "31.73", "60.5"]

MODELS = [
    ("baseline", "StableSR baseline x8"),
    ("controlnet_canny_e3", "ControlNet Canny e3 x8"),
]


def parse_bpp(canny_root: Path, metric: str, lam: str) -> float:
    run_dir = canny_root / metric / f"lambda_{lam}"
    results_json = run_dir / "results.json"
    if results_json.is_file():
        with open(results_json) as f:
            return float(json.load(f)["summary"]["bpp"])

    summary_txt = run_dir / "summary.txt"
    in_avg = False
    for line in summary_txt.read_text().splitlines():
        line = line.strip()
        if line == "[Average]":
            in_avg = True
            continue
        if in_avg and line.startswith("bpp:"):
            return float(line.split(":", 1)[1].strip())
    raise ValueError(f"bpp not found: {run_dir}")


def load_metrics(metrics_dir: Path, model_key: str, tag: str) -> dict:
    path = metrics_dir / f"{model_key}_{tag}_x8.json"
    with open(path) as f:
        return json.load(f)


def collect_points(metrics_dir: Path, canny_root: Path, metric_mode: str = "mse"):
    rows = []
    if metric_mode == "mse":
        groups = [("MSE", MSE_LAMBDAS)]
    elif metric_mode == "msssim":
        groups = [("MSSSIM", MSSSIM_LAMBDAS)]
    elif metric_mode == "both":
        groups = [("MSE", MSE_LAMBDAS), ("MSSSIM", MSSSIM_LAMBDAS)]
    else:
        raise ValueError(f"metric_mode must be mse|msssim|both, got {metric_mode}")
    for metric, lambdas in groups:
        for lam in lambdas:
            tag = f"{metric}_{lam}"
            try:
                bpp = parse_bpp(canny_root, metric, lam)
            except (FileNotFoundError, ValueError) as e:
                print(f"[WARN] skip bpp {tag}: {e}")
                continue
            for model_key, model_label in MODELS:
                metrics_path = metrics_dir / f"{model_key}_{tag}_x8.json"
                if not metrics_path.is_file():
                    print(f"[WARN] skip metrics: {metrics_path}")
                    continue
                m = load_metrics(metrics_dir, model_key, tag)
                rows.append({
                    "model": model_key,
                    "model_label": model_label,
                    "metric": metric,
                    "lambda": lam,
                    "tag": tag,
                    "bpp": bpp,
                    "psnr_y": m["psnr_y_mean"],
                    "ms_ssim_y": m["ms_ssim_y_mean"],
                    "lpips": m["lpips_mean"],
                })
    rows.sort(key=lambda r: r["bpp"])
    return rows


def plot_curve(rows, y_key, ylabel, out_path):
    plt.figure(figsize=(6, 4.5))
    for model_key, model_label in MODELS:
        pts = sorted([r for r in rows if r["model"] == model_key], key=lambda r: r["bpp"])
        if not pts:
            continue
        xs = [p["bpp"] for p in pts]
        ys = [p[y_key] for p in pts]
        plt.plot(xs, ys, "-o", linewidth=2, markersize=6, label=model_label)

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
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--canny-hpcm-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="mse",
        choices=["mse", "msssim", "both"],
        help="Which canny checkpoints to plot (default: mse = 6 points)",
    )
    parser.add_argument(
        "--include-msssim",
        action="store_true",
        help="Deprecated alias for --metric both",
    )
    args = parser.parse_args()

    metric_mode = "both" if args.include_msssim else args.metric

    metrics_dir = Path(args.metrics_dir)
    canny_root = Path(args.canny_hpcm_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_points(metrics_dir, canny_root, metric_mode=metric_mode)

    csv_path = out_dir / "rd_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "model_label", "metric", "lambda", "tag",
                "bpp", "psnr_y", "ms_ssim_y", "lpips",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    plot_curve(rows, "psnr_y", "PSNR", out_dir / "psnr_vs_bpp.png")
    plot_curve(rows, "ms_ssim_y", "MS-SSIM", out_dir / "msssim_vs_bpp.png")
    plot_curve(rows, "lpips", "LPIPS", out_dir / "lpips_vs_bpp.png")


if __name__ == "__main__":
    main()
