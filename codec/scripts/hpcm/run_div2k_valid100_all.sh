#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Run HPCM_Base on DIV2K Valid 100: LR_64 + canny, all 12 rate points (6 MSE + 6 MS-SSIM).
# Total: 24 experiments.
#
# Usage:
#   bash scripts/run_div2k_valid100_all.sh
#   DEVICE=cuda bash scripts/run_div2k_valid100_all.sh
#   SKIP_EXISTING=0 bash scripts/run_div2k_valid100_all.sh   # force re-run

set -euo pipefail

# LIC_ROOT from env.sh

DEVICE="${DEVICE:-cpu}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODEL_NAME="${MODEL_NAME:-HPCM_Base}"
OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100}"

DATASETS=(
  "lr64|/data/Dataset/DIV2K/DIV2K_valid_100_512_64/LR_64"
  "canny|/data/Dataset/DIV2K/DIV2K_valid_100_512_128/canny"
)

MSE_LAMBDAS=(0.0018 0.0035 0.0067 0.013 0.025 0.0483)
MSSSIM_LAMBDAS=(2.4 4.58 8.73 16.64 31.73 60.5)

LOG_DIR="${OUT_ROOT}/logs"
MASTER_CSV="${OUT_ROOT}/all_runs_summary.csv"
mkdir -p "${LOG_DIR}"

echo "LIC_ROOT=${LIC_ROOT}"
echo "DEVICE=${DEVICE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo ""

run_one() {
  local dataset_tag="$1"
  local dataset_path="$2"
  local metric="$3"    # MSE or MSSSIM
  local lambda="$4"
  local ckpt="${LIC_ROOT}/checkpoints/HPCM_Base/${metric}/${lambda}.pth.tar"

  local run_dir="${OUT_ROOT}/${dataset_tag}/${metric}/lambda_${lambda}"
  local img_dir="${run_dir}/images"
  local results_json="${run_dir}/results.json"
  local log_file="${LOG_DIR}/${dataset_tag}_${metric}_lambda${lambda}.log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[SKIP] missing checkpoint: ${ckpt}"
    return 1
  fi
  if [[ ! -d "${dataset_path}" ]]; then
    echo "[SKIP] missing dataset: ${dataset_path}"
    return 1
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${results_json}" ]]; then
    echo "[SKIP] done: ${run_dir}"
    return 0
  fi

  echo "========================================"
  echo "[RUN] ${dataset_tag} | ${metric} | lambda=${lambda}"
  echo "  dataset: ${dataset_path}"
  echo "  ckpt:    ${ckpt}"
  echo "  out:     ${run_dir}"
  echo "========================================"

  mkdir -p "${img_dir}" "${run_dir}"
  cd "${CODEC_ROOT}"

  "${PYTHON}" test.py \
    --model_name "${MODEL_NAME}" \
    --dataset "${dataset_path}" \
    --checkpoint "${ckpt}" \
    --outdir "${img_dir}" \
    --results_dir "${run_dir}" \
    --device "${DEVICE}" \
    2>&1 | tee "${log_file}"
}

# --- main ---
total=0
ok=0
fail=0

for entry in "${DATASETS[@]}"; do
  dataset_tag="${entry%%|*}"
  dataset_path="${entry#*|}"

  for lambda in "${MSE_LAMBDAS[@]}"; do
    total=$((total + 1))
    if run_one "${dataset_tag}" "${dataset_path}" "MSE" "${lambda}"; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
    fi
  done

  for lambda in "${MSSSIM_LAMBDAS[@]}"; do
    total=$((total + 1))
    if run_one "${dataset_tag}" "${dataset_path}" "MSSSIM" "${lambda}"; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
    fi
  done
done

# Aggregate all per-run results.json into one CSV
echo ""
echo "Building master summary: ${MASTER_CSV}"
"${PYTHON}" - <<PY
import csv, json, os, glob

out_root = "${OUT_ROOT}"
master = os.path.join(out_root, "all_runs_summary.csv")
rows = []
for path in sorted(glob.glob(os.path.join(out_root, "*", "MSE", "lambda_*", "results.json"))):
    rows.append(path)
for path in sorted(glob.glob(os.path.join(out_root, "*", "MSSSIM", "lambda_*", "results.json"))):
    rows.append(path)

fieldnames = [
    "dataset", "metric", "lambda", "checkpoint", "device", "num_images",
    "psnr", "msssim_db", "msssim_metric", "bpp", "y_bpp", "z_bpp",
    "enc_time", "dec_time", "results_dir",
]
out_rows = []
for path in rows:
    with open(path) as f:
        d = json.load(f)
    parts = path.replace(out_root + os.sep, "").split(os.sep)
    dataset_tag, metric, lambda_dir = parts[0], parts[1], parts[2]
    lam = lambda_dir.replace("lambda_", "")
    s = d["summary"]
    out_rows.append({
        "dataset": dataset_tag,
        "metric": metric,
        "lambda": lam,
        "checkpoint": d["checkpoint"],
        "device": d["device"],
        "num_images": d["num_images"],
        "psnr": s["psnr"],
        "msssim_db": s["msssim_db"],
        "msssim_metric": s.get("msssim_metric", ""),
        "bpp": s["bpp"],
        "y_bpp": s["y_bpp"],
        "z_bpp": s["z_bpp"],
        "enc_time": s["enc_time"],
        "dec_time": s["dec_time"],
        "results_dir": os.path.dirname(path),
    })

os.makedirs(out_root, exist_ok=True)
with open(master, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)
print(f"Wrote {len(out_rows)} rows -> {master}")
PY

echo ""
echo "Done. total=${total} ok=${ok} fail=${fail}"
echo "Master CSV: ${MASTER_CSV}"
echo "Per-run logs: ${LOG_DIR}/"
