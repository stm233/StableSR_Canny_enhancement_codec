#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Stage 2 (cached): P-frame DT1ch training with offline I-frame ref cache.
#
# Run prepare_pframe_ref_cache.sh first with the SAME INIT_CKPT.
#
# Usage:
#   INIT_CKPT=.../epoch_best.pth.tar \
#   PFRAME_CACHE_ROOT=.../pframe_ref_cache \
#   bash codec/scripts/hpcm/train_hpcm_video_pframe_dt_cached.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_processed}"
INIT_CKPT="${INIT_CKPT:-}"
PFRAME_CACHE_ROOT="${PFRAME_CACHE_ROOT:-}"
PFRAME_RESUME="${PFRAME_RESUME:-}"
P_CODEC_INIT="${P_CODEC_INIT:-}"
LAMBDA="${LAMBDA:-0.00105}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATCH_SIZE="${PATCH_SIZE:-256}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/video_pframe_dt_cached_lambda0.00105}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

[[ -f "${DATA_ROOT}/manifest_pframe.jsonl" ]] || {
  echo "Missing ${DATA_ROOT}/manifest_pframe.jsonl"
  exit 1
}
[[ -n "${PFRAME_CACHE_ROOT}" ]] || { echo "Set PFRAME_CACHE_ROOT=... (from prepare_pframe_ref_cache.sh)"; exit 1; }
[[ -f "${PFRAME_CACHE_ROOT}/cache_meta.json" ]] || {
  echo "Missing cache meta: ${PFRAME_CACHE_ROOT}/cache_meta.json"
  echo "Run: bash codec/scripts/hpcm/prepare_pframe_ref_cache.sh"
  exit 1
}

ckpt_args=()
resume_args=()
if [[ -n "${PFRAME_RESUME}" ]]; then
  [[ -f "${PFRAME_RESUME}" ]] || { echo "Missing PFRAME_RESUME: ${PFRAME_RESUME}"; exit 1; }
  resume_args=(--resume "${PFRAME_RESUME}")
  echo "P-frame resume:          ${PFRAME_RESUME}"
elif [[ -n "${INIT_CKPT}" ]]; then
  [[ -f "${INIT_CKPT}" ]] || { echo "Missing I-frame ckpt: ${INIT_CKPT}"; exit 1; }
  ckpt_args=(--checkpoint "${INIT_CKPT}")
  echo "I-frame DT ckpt (model): ${INIT_CKPT}"
else
  echo "I-frame ckpt: (none)"
fi

p_codec_args=()
if [[ -z "${PFRAME_RESUME}" && -n "${P_CODEC_INIT}" ]]; then
  [[ -f "${P_CODEC_INIT}" ]] || { echo "Missing P_CODEC_INIT: ${P_CODEC_INIT}"; exit 1; }
  p_codec_args=(--p-codec-init "${P_CODEC_INIT}")
  echo "P-codec init:            ${P_CODEC_INIT}"
fi

echo "Stage:      pframe (cached refs)"
echo "Model:      HPCM_Video_PFrame_DT1ch"
echo "Lambda:     ${LAMBDA}"
echo "Data:       ${DATA_ROOT}"
echo "Ref cache:  ${PFRAME_CACHE_ROOT}"
echo "Save:       ${SAVE_PATH}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" train_video.py \
  --stage pframe \
  --model_name HPCM_Video_PFrame_DT1ch \
  --dataset-root "${DATA_ROOT}" \
  --pframe-cache-dir "${PFRAME_CACHE_ROOT}" \
  "${ckpt_args[@]}" \
  "${resume_args[@]}" \
  "${p_codec_args[@]}" \
  --lambda "${LAMBDA}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --patch-size "${PATCH_SIZE}" "${PATCH_SIZE}" \
  --save_path "${SAVE_PATH}" \
  --log_dir "${LOG_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --clip_max_norm 1.0 \
  2>&1 | tee -a "${OUT_ROOT}/train.log"
