#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Stage 1: I-frame codec — single-channel binary Canny [1,H,W].
#
# Usage:
#   bash scripts/train_hpcm_video_iframe.sh
#   PREPARE=1 bash scripts/train_hpcm_video_iframe.sh
#   INIT_CKPT=checkpoints/HPCM_Base/MSE/0.0018.pth.tar bash scripts/train_hpcm_video_iframe.sh

set -euo pipefail

# LIC_ROOT from env.sh


VIDEO_ROOT="${VIDEO_ROOT:-/data/Dataset/HQ-VSR}"
DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_processed}"
PREPARE="${PREPARE:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
FRAME_STEP="${FRAME_STEP:-1}"
SIZE="${SIZE:-512}"

INIT_CKPT="${INIT_CKPT:-}"
LAMBDA="${LAMBDA:-0.00105}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATCH_SIZE="${PATCH_SIZE:-256}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/video_iframe_canny1ch_lambda0.00105}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

if [[ "${PREPARE}" == "1" ]]; then
  extra=()
  if [[ "${MAX_VIDEOS}" != "0" ]]; then
    extra+=(--max-videos "${MAX_VIDEOS}")
  fi
  "${PYTHON}" "${CODEC_ROOT}/scripts/hpcm/prepare_hqvsr_video_dataset.py" \
    --video-root "${VIDEO_ROOT}" \
    --out-dir "${DATA_ROOT}" \
    --size "${SIZE}" \
    --frame-step "${FRAME_STEP}" \
    "${extra[@]}"
fi

[[ -f "${DATA_ROOT}/manifest_iframe.jsonl" ]] || {
  echo "Missing ${DATA_ROOT}/manifest_iframe.jsonl — run with PREPARE=1 first"
  exit 1
}

ckpt_args=()
if [[ -n "${INIT_CKPT}" ]]; then
  [[ -f "${INIT_CKPT}" ]] || { echo "Missing init ckpt: ${INIT_CKPT}"; exit 1; }
  ckpt_args=(--checkpoint "${INIT_CKPT}")
  echo "Init ckpt:  ${INIT_CKPT}"
else
  echo "Init ckpt:  (none, from scratch)"
fi

echo "Stage:      iframe (1ch canny)"
echo "Lambda:     ${LAMBDA}"
echo "Save every: ${SAVE_INTERVAL} epochs"
echo "Data:       ${DATA_ROOT}"
echo "Save:       ${SAVE_PATH}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" train_video.py \
  --stage iframe \
  --model_name HPCM_Canny1ch \
  --dataset-root "${DATA_ROOT}" \
  "${ckpt_args[@]}" \
  --lambda "${LAMBDA}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --patch-size "${PATCH_SIZE}" "${PATCH_SIZE}" \
  --save_path "${SAVE_PATH}" \
  --log_dir "${LOG_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --clip_max_norm 1.0 \
  2>&1 | tee "${OUT_ROOT}/train.log"
