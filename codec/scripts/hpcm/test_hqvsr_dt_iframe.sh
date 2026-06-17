#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Test HPCM_DT1ch on HQ-VSR_test500.
# Pipeline: Canny -> DT 3ch in -> decode R_hat -> binary Canny (recon/).

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_test500}"
CHECKPOINT="${CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
DIST_TOL="${DIST_TOL:-0.5}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/HQ-VSR_test500_dt_iframe}"
RESULTS_DIR="${OUT_ROOT}/metrics"
IMG_DIR="${OUT_ROOT}/images"

[[ -f "${DATA_ROOT}/manifest_iframe.jsonl" ]] || {
  echo "Missing ${DATA_ROOT}/manifest_iframe.jsonl"
  exit 1
}
[[ -n "${CHECKPOINT}" ]] || { echo "Set CHECKPOINT=... (HPCM_DT1ch ckpt)"; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}"; exit 1; }

mkdir -p "${RESULTS_DIR}"
extra=()
[[ "${SAVE_IMAGES}" == "1" ]] && extra+=(--outdir "${IMG_DIR}") && mkdir -p "${IMG_DIR}"

echo "Model:      HPCM_DT1ch"
echo "Test set:   ${DATA_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "dist_tol:   ${DIST_TOL} px (R_hat*(H+W) <= tol -> edge 255)"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" test_video_iframe.py \
  --model_name HPCM_DT1ch \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --manifest manifest_iframe.jsonl \
  --device "${DEVICE}" \
  --dist-tol "${DIST_TOL}" \
  --results_dir "${RESULTS_DIR}" \
  "${extra[@]}" \
  2>&1 | tee "${OUT_ROOT}/test.log"

echo "Results: ${RESULTS_DIR}/results.json"
[[ "${SAVE_IMAGES}" == "1" ]] && echo "recon/     binary Canny (from R_hat post-process)"
[[ "${SAVE_IMAGES}" == "1" ]] && echo "recon_raw/ R_hat normalized + *_dist_px.png debug"
