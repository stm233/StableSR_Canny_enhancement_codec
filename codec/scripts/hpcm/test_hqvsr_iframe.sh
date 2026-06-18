#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Test I-frame on fixed HQ-VSR_test500 (500 consecutive Canny frames).
#
# Usage:
#   bash scripts/test_hqvsr_iframe.sh
#   DEVICE=cpu bash scripts/test_hqvsr_iframe.sh

set -euo pipefail

# LIC_ROOT from env.sh


DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_test500}"
MODEL_NAME="${MODEL_NAME:-HPCM_Canny1ch}"
CHECKPOINT="${CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/HQ-VSR_test500_iframe}"
RESULTS_DIR="${OUT_ROOT}/metrics"
IMG_DIR="${OUT_ROOT}/images"

[[ -f "${DATA_ROOT}/manifest_iframe.jsonl" ]] || {
  echo "Missing ${DATA_ROOT}. Run: python scripts/create_hqvsr_test500.py"
  exit 1
}

[[ -n "${CHECKPOINT}" ]] || { echo "Set CHECKPOINT=... (HPCM_Canny1ch ckpt)"; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}"; exit 1; }

mkdir -p "${RESULTS_DIR}"
extra=()
[[ "${SAVE_IMAGES}" == "1" ]] && extra+=(--outdir "${IMG_DIR}") && mkdir -p "${IMG_DIR}"

echo "Model:      ${MODEL_NAME}"
echo "Test set:   ${DATA_ROOT} (500 fixed I-frames)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Device:     ${DEVICE}"
echo ""

cd "${CODEC_ROOT}"

"${PYTHON}" test_video_iframe.py \
  --model_name "${MODEL_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --manifest manifest_iframe.jsonl \
  --device "${DEVICE}" \
  --results_dir "${RESULTS_DIR}" \
  "${extra[@]}" \
  2>&1 | tee "${OUT_ROOT}/test.log"

echo "Results: ${RESULTS_DIR}/results.json"
[[ "${SAVE_IMAGES}" == "1" ]] && echo "Visuals:  ${IMG_DIR}/gt  ${IMG_DIR}/recon  ${IMG_DIR}/compare"
