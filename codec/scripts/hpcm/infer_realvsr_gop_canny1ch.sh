#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# GOP infer on RealVSR (Canny1ch): I + N×P bitstream per video.
#
# Quick smoke test (1 video, 1 GOP = I+P):
#   NUM_P=1 MAX_VIDEOS=1 bash codec/scripts/hpcm/infer_realvsr_gop_canny1ch.sh
#
# Full RealVSR test set:
#   bash codec/scripts/hpcm/infer_realvsr_gop_canny1ch.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset/RealVSR_GT_test_iframe_all}"
PFRAME_CKPT="${PFRAME_CKPT:-/home/exx/Documents/Tianma/StableSR/codec/checkpoints/P_frame/lambda_11_epoch_7.tar}"
IFRAME_CKPT="${IFRAME_CKPT:-}"
NUM_P="${NUM_P:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_GOPS_PER_VIDEO="${MAX_GOPS_PER_VIDEO:-1}"
DEVICE="${DEVICE:-cuda}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/RealVSR_gop_canny1ch_test}"
RESULTS_DIR="${RESULTS_DIR:-${OUT_ROOT}/metrics}"
IMG_DIR="${IMG_DIR:-${OUT_ROOT}/images}"
# SAVE_IMAGES=0: metrics only, no recon PNG export (faster, less disk)
SAVE_IMAGES="${SAVE_IMAGES:-1}"
# EXPORT_STABLESR=0: skip outdir/canny/images/ (StableSR naming); only if SAVE_IMAGES=1
EXPORT_STABLESR="${EXPORT_STABLESR:-1}"

[[ -d "${DATA_ROOT}/canny" ]] || {
  echo "Missing ${DATA_ROOT}/canny"
  exit 1
}
[[ -f "${PFRAME_CKPT}" ]] || {
  echo "Missing P-frame ckpt: ${PFRAME_CKPT}"
  exit 1
}

mkdir -p "${RESULTS_DIR}"
[[ "${SAVE_IMAGES}" == "1" ]] && mkdir -p "${IMG_DIR}"

extra=()
[[ -n "${IFRAME_CKPT}" ]] && extra+=(--iframe-checkpoint "${IFRAME_CKPT}")
[[ "${SAVE_IMAGES}" == "1" ]] && extra+=(--outdir "${OUT_ROOT}")
[[ "${EXPORT_STABLESR}" == "0" ]] && extra+=(--no-export-stablesr)

echo "Model:      HPCM_Video_PFrame_Canny1ch"
echo "Dataset:    ${DATA_ROOT}"
echo "GOP:        I + ${NUM_P}P"
echo "Checkpoint: ${PFRAME_CKPT}"
echo "Max videos: ${MAX_VIDEOS} (0=all)"
echo "Max GOPs/video: ${MAX_GOPS_PER_VIDEO} (0=all)"
echo "Save images:    ${SAVE_IMAGES} (0=metrics only)"
echo "Export StableSR canny layout: ${EXPORT_STABLESR}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" infer_video_gop.py \
  --model-name HPCM_Video_PFrame_Canny1ch \
  --pframe-checkpoint "${PFRAME_CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --num-p "${NUM_P}" \
  --max-videos "${MAX_VIDEOS}" \
  --max-gops-per-video "${MAX_GOPS_PER_VIDEO}" \
  --device "${DEVICE}" \
  --results-dir "${RESULTS_DIR}" \
  "${extra[@]}" \
  2>&1 | tee "${OUT_ROOT}/infer.log"

echo "Results: ${RESULTS_DIR}/"
if [[ "${SAVE_IMAGES}" == "1" ]]; then
  if [[ "${EXPORT_STABLESR}" == "1" ]]; then
    echo "Canny (StableSR --canny-dir): ${OUT_ROOT}/canny/images/"
    echo "  naming: {video}_{frame}.png  e.g. 016_00000.png"
    echo "Pair with LQ: RealVSR_GT_test_lr64/MSE/lambda_*/images/ (HPCM_Base --init-img)"
    echo "Pair with GT: RealVSR_GT_test_iframe_all_flat/gt512/ (--gt-img)"
  else
    echo "Images: ${OUT_ROOT}/  (legacy 016_f000000_I.png naming)"
  fi
else
  echo "Images: not saved (SAVE_IMAGES=0). Re-run with SAVE_IMAGES=1 to export canny for StableSR."
fi
