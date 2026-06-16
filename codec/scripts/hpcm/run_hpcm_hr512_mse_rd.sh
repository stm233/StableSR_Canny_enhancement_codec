#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# HPCM_Base: compress DIV2K Valid100 HR 512x512 directly (6 MSE rate points).
# Then evaluate PSNR(Y)/MS-SSIM(Y)/LPIPS vs GT and plot RD curves.
#
# Usage:
#   bash scripts/run_hpcm_hr512_mse_rd.sh
#   DEVICE=cuda bash scripts/run_hpcm_hr512_mse_rd.sh
#   RUN_COMPRESS=0 bash scripts/run_hpcm_hr512_mse_rd.sh   # eval + plot only

set -euo pipefail

# LIC_ROOT from env.sh
STABLESR_ROOT="${STABLESR_ROOT:-/home/exx/Documents/Tianma/StableSR}"


DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_COMPRESS="${RUN_COMPRESS:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

DATASET="/data/Dataset/DIV2K/DIV2K_valid_100_512_128/HR_512"
OUT_ROOT="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/hr512/MSE"
METRICS_DIR="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/hr512/rd_metrics"
CURVE_DIR="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/hr512/rd_curves"
LOG_DIR="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/hr512/logs"

# low -> high bpp
LAMBDAS=(0.0018 0.0035 0.0067 0.013 0.025 0.0483)

mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${CURVE_DIR}"

echo "LIC_ROOT=${LIC_ROOT}"
echo "DATASET=${DATASET}"
echo "DEVICE=${DEVICE}"
echo ""

run_one() {
  local lam="$1"
  local ckpt="${LIC_ROOT}/checkpoints/HPCM_Base/MSE/${lam}.pth.tar"
  local run_dir="${OUT_ROOT}/lambda_${lam}"
  local img_dir="${run_dir}/images"
  local results_json="${run_dir}/results.json"
  local metrics_json="${METRICS_DIR}/lambda_${lam}.json"
  local log="${LOG_DIR}/hr512_MSE_${lam}.log"

  [[ -f "${ckpt}" ]] || { echo "[SKIP] missing ckpt: ${ckpt}"; return 1; }

  if [[ "${RUN_COMPRESS}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -f "${results_json}" ]]; then
      echo "[SKIP compress] lambda=${lam}"
    else
      echo "======== HPCM HR512 | MSE lambda=${lam} ========"
      mkdir -p "${img_dir}"
      cd "${CODEC_ROOT}"
      "${PYTHON}" test.py \
        --model_name HPCM_Base \
        --dataset "${DATASET}" \
        --checkpoint "${ckpt}" \
        --outdir "${img_dir}" \
        --results_dir "${run_dir}" \
        --device "${DEVICE}" \
        2>&1 | tee "${log}"
    fi
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics_json}" ]]; then
      echo "[SKIP eval] ${metrics_json}"
    else
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/eval_sr_metrics.py \
        --gt-dir "${DATASET}" \
        --out-dir "${img_dir}" \
        --device "${DEVICE}" \
        --skip-fid \
        --json-out "${metrics_json}"
    fi
  fi
}

for lam in "${LAMBDAS[@]}"; do
  run_one "${lam}"
done

if [[ "${RUN_PLOT}" == "1" ]]; then
  cd "${STABLESR_ROOT}"
  SR_METRICS="/data/Dataset/StableSR-TestSets/codec_rd/x8/metrics"
  SR_HPCM_LR64="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/lr64/MSE"
  COMBINED_OUT="/data/Dataset/StableSR-TestSets/codec_rd/x8/rd_curves"
  if [[ -d "${SR_METRICS}" ]]; then
    "${PYTHON}" scripts/plot_hpcm_lq_sr_rd_curves.py \
      --metrics-dir "${SR_METRICS}" \
      --hpcm-root "${SR_HPCM_LR64}" \
      --hpcm-hr512-root "${OUT_ROOT}" \
      --hpcm-hr512-metrics-dir "${METRICS_DIR}" \
      --out-dir "${COMBINED_OUT}"
  else
    "${PYTHON}" scripts/plot_hpcm_lq_sr_rd_curves.py \
      --hpcm-hr512-root "${OUT_ROOT}" \
      --hpcm-hr512-metrics-dir "${METRICS_DIR}" \
      --out-dir "${CURVE_DIR}"
  fi
fi

echo ""
echo "Done."
echo "Compression: ${OUT_ROOT}/lambda_*/"
echo "Metrics:     ${METRICS_DIR}/"
echo "Curves:      ${CURVE_DIR}/"
