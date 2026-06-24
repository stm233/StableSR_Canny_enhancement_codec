#!/usr/bin/env bash
# HPCM_Base compress/decompress RealVSR LR64 (flat folder from prepare_realvsr_lr64_flat.py).
#
# Checkpoints: codec/checkpoints/MSE/{lambda}.pth.tar  (same as HPCM_Base MSE weights)
#
# Usage:
#   bash codec/scripts/hpcm/prepare_realvsr_lr64_flat.sh   # once
#   LAMBDA=0.0035 bash codec/scripts/hpcm/test_realvsr_hpcm_base.sh
#   METRIC=all LAMBDA=0.0035 bash ...   # all 6 MSE lambdas

source "$(dirname "$0")/env.sh"
set -euo pipefail

DATA_FLAT="${DATA_FLAT:-/data/Dataset/RealVSR_GT_test_iframe_all_flat}"
LR_DIR="${DATA_FLAT}/lr64"
OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/RealVSR_GT_test_lr64}"
LAMBDA="${LAMBDA:-0.0035}"
DEVICE="${DEVICE:-cuda}"

CKPT_DIR="${CODEC_ROOT}/checkpoints/MSE"
MSE_LAMBDAS=(0.0018 0.0035 0.0067 0.013 0.025 0.0483)

[[ -d "${LR_DIR}" ]] || {
  echo "Missing ${LR_DIR}. Run: bash codec/scripts/hpcm/prepare_realvsr_lr64_flat.sh"
  exit 1
}

run_one() {
  local lam="$1"
  local ckpt="${CKPT_DIR}/${lam}.pth.tar"
  local run_dir="${OUT_ROOT}/MSE/lambda_${lam}"
  local img_dir="${run_dir}/images"

  [[ -f "${ckpt}" ]] || { echo "[SKIP] no ckpt: ${ckpt}"; return 1; }
  mkdir -p "${img_dir}" "${run_dir}"

  echo "======== HPCM_Base MSE lambda=${lam} ========"
  cd "${CODEC_ROOT}"
  "${PYTHON}" test.py \
    --model_name HPCM_Base \
    --dataset "${LR_DIR}" \
    --checkpoint "${ckpt}" \
    --outdir "${img_dir}" \
    --results_dir "${run_dir}" \
    --device "${DEVICE}"
}

if [[ "${LAMBDA}" == "all" ]]; then
  for lam in "${MSE_LAMBDAS[@]}"; do
    run_one "${lam}"
  done
else
  run_one "${LAMBDA}"
fi

echo "Done. Decompressed LQ: ${OUT_ROOT}/MSE/lambda_*/images/"
