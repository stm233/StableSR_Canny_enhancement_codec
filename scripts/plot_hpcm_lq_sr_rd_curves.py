#!/usr/bin/env python3
"""Plot combined R-D curves: SR x8 + optional HPCM direct 512x512 baseline."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


LAMBDA_ORDER = ["0.0018", "0.0035", "0.0067", "0.013", "0.025", "0.0483"]

SR_MODELS = [
    ("baseline", "StableSR baseline x8"),
    ("controlnet_canny_e3", "ControlNet Canny e3 x8"),
]
HPCM_HR512_MODEL = ("hpcm_hr512", "HPCM")


def parse_bpp_from_results(hpcm_root: Path, lam: str) -> float:
    results_json = hpcm_root / f"lambda_{lam}" / "results.json"
    if results_json.is_file():
        with open(results_json) as f:
            return float(json.load(f)["summary"]["bpp"])

    summary_txt = hpcm_root / f"lambda_{lam}" / "summary.txt"
    if not summary_txt.is_file():
        raise FileNotFoundError(f"No HPCM summary for lambda={lam}: {summary_txt}")

    in_avg = False
    for line in summary_txt.read_text().splitlines():
        line = line.strip()
        if line == "[Average]":
            in_avg = True
            continue
        if in_avg and line.startswith("bpp:"):
            return float(line.split(":", 1)[1].strip())
        if in_avg and line.startswith("[") and line.endswith("]"):
            break
    raise ValueError(f"bpp not found in {summary_txt}")


def load_sr_metrics(metrics_dir: Path, model_key: str, lam: str) -> dict:
    path = metrics_dir / f"{model_key}_MSE_{lam}_x8.json"
    with open(path) as f:
        return json.load(f)


def load_hpcm_hr512_metrics(metrics_dir: Path, lam: str) -> dict:
    path = metrics_dir / f"lambda_{lam}.json"
    with open(path) as f:
        return json.load(f)


def collect_sr_rows(metrics_dir: Path, hpcm_lr64_root: Path, scale_bpp: float):
    rows = []
    for model_key, model_label in SR_MODELS:
        for lam in LAMBDA_ORDER:
            metrics_path = metrics_dir / f"{model_key}_MSE_{lam}_x8.json"
            if not metrics_path.is_file():
                print(f"[WARN] skip SR: missing {metrics_path}")
                continue
            try:
                bpp_64 = parse_bpp_from_results(hpcm_lr64_root, lam)
                m = load_sr_metrics(metrics_dir, model_key, lam)
            except (FileNotFoundError, ValueError) as e:
                print(f"[WARN] skip SR {model_key} lambda={lam}: {e}")
                continue
            rows.append({
                "model": model_key,
                "model_label": model_label,
                "lambda": lam,
                "bpp": bpp_64 / scale_bpp,
                "psnr_y": m["psnr_y_mean"],
                "ms_ssim_y": m["ms_ssim_y_mean"],
                "lpips": m["lpips_mean"],
            })
    return rows


def collect_hpcm_hr512_rows(hpcm_hr512_root: Path, metrics_dir: Path):
    rows = []
    model_key, model_label = HPCM_HR512_MODEL
    for lam in LAMBDA_ORDER:
        metrics_path = metrics_dir / f"lambda_{lam}.json"
        if not metrics_path.is_file():
            print(f"[WARN] skip HPCM HR512: missing {metrics_path}")
            continue
        try:
            m = load_hpcm_hr512_metrics(metrics_dir, lam)
            bpp = parse_bpp_from_results(hpcm_hr512_root, lam)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] skip HPCM HR512 lambda={lam}: {e}")
            continue
        rows.append({
            "model": model_key,
            "model_label": model_label,
            "lambda": lam,
            "bpp": bpp,
            "psnr_y": m["psnr_y_mean"],
            "ms_ssim_y": m["ms_ssim_y_mean"],
            "lpips": m["lpips_mean"],
        })
    return rows


def plot_curve(rows, plot_models, y_key, ylabel, out_path):
    plt.figure(figsize=(6, 4.5))
    for model_key, model_label in plot_models:
        pts = [r for r in rows if r["model"] == model_key]
        pts = sorted(pts, key=lambda r: r["bpp"])
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
    parser.add_argument("--metrics-dir", type=str, default="",
                        help="SR x8 metrics JSON dir")
    parser.add_argument("--hpcm-root", type=str, default="",
                        help="HPCM lr64 MSE results (for bpp@64 -> bpp@512)")
    parser.add_argument("--hpcm-hr512-root", type=str, default="",
                        help="HPCM direct 512x512 MSE results root")
    parser.add_argument("--hpcm-hr512-metrics-dir", type=str, default="",
                        help="HPCM 512x512 eval metrics JSON dir")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--scale-bpp",
        type=float,
        default=64.0,
        help="SR path: bpp@512 = bpp@64 / scale_bpp",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    plot_models = []

    if args.metrics_dir and args.hpcm_root:
        sr_rows = collect_sr_rows(
            Path(args.metrics_dir), Path(args.hpcm_root), args.scale_bpp
        )
        if sr_rows:
            rows.extend(sr_rows)
            plot_models.extend(SR_MODELS)

    if args.hpcm_hr512_root and args.hpcm_hr512_metrics_dir:
        hr512_rows = collect_hpcm_hr512_rows(
            Path(args.hpcm_hr512_root), Path(args.hpcm_hr512_metrics_dir)
        )
        if hr512_rows:
            rows.extend(hr512_rows)
            if HPCM_HR512_MODEL not in plot_models:
                plot_models.append(HPCM_HR512_MODEL)

    if not rows:
        raise SystemExit("No data: provide SR paths and/or HPCM HR512 paths.")

    csv_path = out_dir / "rd_summary_combined.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "model_label", "lambda", "bpp",
                "psnr_y", "ms_ssim_y", "lpips",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    plot_curve(rows, plot_models, "psnr_y", "PSNR", out_dir / "psnr_vs_bpp.png")
    plot_curve(rows, plot_models, "ms_ssim_y", "MS-SSIM", out_dir / "msssim_vs_bpp.png")
    plot_curve(rows, plot_models, "lpips", "LPIPS", out_dir / "lpips_vs_bpp.png")


if __name__ == "__main__":
    main()
