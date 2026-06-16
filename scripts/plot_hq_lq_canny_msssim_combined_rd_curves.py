#!/usr/bin/env python3
"""Plot combined LQ+canny bitrate curves (hq LQ + MS-SSIM canny)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MSSSIM_LAMBDAS = ["2.4", "4.58", "8.73", "16.64", "31.73", "60.5"]

BASELINE_KEY = "baseline"
BASELINE_LABEL = "StableSR baseline x8 (LQ only)"
CN_KEY = "controlnet_canny_e3"
CN_LABEL = "ControlNet Canny e3 x8"


def parse_bpp_from_run(run_dir: Path) -> float:
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
    raise ValueError(f"bpp not found in {run_dir}")


def load_metrics(metrics_dir: Path, name: str) -> dict:
    with open(metrics_dir / f"{name}.json") as f:
        return json.load(f)


def collect_points(metrics_dir: Path, lq_summary_dir: Path, canny_root: Path, lq_scale: float):
    lq_bpp_64 = parse_bpp_from_run(lq_summary_dir)
    lq_bpp_512 = lq_bpp_64 / lq_scale

    rows = []

    m = load_metrics(metrics_dir, "baseline_LQ_MSE_0.0483_x8")
    rows.append({
        "model": BASELINE_KEY,
        "model_label": BASELINE_LABEL,
        "tag": "LQ_MSE_0.0483",
        "lambda": "0.0483",
        "lq_bpp_64": lq_bpp_64,
        "lq_bpp_512": lq_bpp_512,
        "canny_bpp": 0.0,
        "bpp": lq_bpp_512,
        "psnr_y": m["psnr_y_mean"],
        "ms_ssim_y": m["ms_ssim_y_mean"],
        "lpips": m["lpips_mean"],
    })

    for lam in MSSSIM_LAMBDAS:
        tag = f"MSSSIM_{lam}"
        canny_bpp = parse_bpp_from_run(canny_root / f"lambda_{lam}")
        total_bpp = lq_bpp_512 + canny_bpp
        m = load_metrics(metrics_dir, f"controlnet_canny_e3_{tag}_x8")
        rows.append({
            "model": CN_KEY,
            "model_label": CN_LABEL,
            "tag": tag,
            "lambda": lam,
            "lq_bpp_64": lq_bpp_64,
            "lq_bpp_512": lq_bpp_512,
            "canny_bpp": canny_bpp,
            "bpp": total_bpp,
            "psnr_y": m["psnr_y_mean"],
            "ms_ssim_y": m["ms_ssim_y_mean"],
            "lpips": m["lpips_mean"],
        })

    return rows


def plot_curve(rows, y_key, ylabel, out_path):
    baseline = [r for r in rows if r["model"] == BASELINE_KEY][0]
    cn_pts = sorted([r for r in rows if r["model"] == CN_KEY], key=lambda r: r["bpp"])

    fig, ax = plt.subplots(figsize=(6, 4.5))

    if cn_pts:
        y_val = baseline[y_key]
        xs = [p["bpp"] for p in cn_pts]
        ys = [p[y_key] for p in cn_pts]
        ax.plot(xs, ys, "-o", linewidth=2, markersize=6, color="C1", label=CN_LABEL)

        x_min = min(xs + [baseline["bpp"]])
        x_max = max(xs)
        pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.01
        ax.set_xlim(x_min - pad, x_max + pad)

        ax.axhline(
            y_val, linestyle="--", linewidth=2, color="C0",
            label=BASELINE_LABEL, zorder=0,
        )
        ax.plot(
            [baseline["bpp"]], [y_val],
            "o", linewidth=2, markersize=6, color="C0", zorder=3,
        )

    ax.set_xlabel("BPP")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--lq-hpcm-summary", type=str, required=True)
    parser.add_argument("--canny-hpcm-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--lq-bpp-scale", type=float, default=64.0)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_points(
        metrics_dir,
        Path(args.lq_hpcm_summary),
        Path(args.canny_hpcm_root),
        args.lq_bpp_scale,
    )

    csv_path = out_dir / "rd_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "model_label", "tag", "lambda",
                "lq_bpp_64", "lq_bpp_512", "canny_bpp", "bpp",
                "psnr_y", "ms_ssim_y", "lpips",
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
