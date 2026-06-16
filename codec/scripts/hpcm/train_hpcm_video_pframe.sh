#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Stage 2: P-frame codec with ref I-frame encoder features fused at decoder input.
#
# INIT_CKPT = I-frame / canny codec ONLY (frozen iframe_codec + ref_encoder init).
#             Does NOT load into P-frame main codec.
# P-frame codec (self.codec) trains from scratch unless P_CODEC_INIT is set.
#
# No offline I-frame compression needed — preprocessing already has canny/gray PNGs.
#
# Usage:
#   bash scripts/train_hpcm_video_pframe.sh
#   P_CODEC_INIT=checkpoints/HPCM_Base/MSE/0.0018.pth.tar bash scripts/train_hpcm_video_pframe.sh

set -euo pipefail

# LIC_ROOT from env.sh


DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_processed}"

# I-frame codec (your trained canny model) — ref path only, NOT the P-frame model.
INIT_CKPT="${INIT_CKPT:-/data/Dataset/LIC-HPCM_outputs/train_canny_lambda0.00105/checkpoints/HPCM_Base_lmbda0.00105/epoch_best.pth.tar}"
# PFRAME_RESUME = full P-frame ckpt (resume). If set, INIT_CKPT is ignored.
PFRAME_RESUME="${PFRAME_RESUME:-}"
P_CODEC_INIT="${P_CODEC_INIT:-}"
LAMBDA="${LAMBDA:-0.00105}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATCH_SIZE="${PATCH_SIZE:-256}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/video_pframe_lambda0.00105}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

[[ -f "${DATA_ROOT}/manifest_pframe.jsonl" ]] || {
  echo "Missing ${DATA_ROOT}/manifest_pframe.jsonl — run prepare script first"
  exit 1
}

ckpt_args=()
resume_args=()
if [[ -n "${PFRAME_RESUME}" ]]; then
  [[ -f "${PFRAME_RESUME}" ]] || {
    echo "Missing PFRAME_RESUME: ${PFRAME_RESUME}"
    exit 1
  }
  resume_args=(--resume "${PFRAME_RESUME}")
  echo "P-frame resume:          ${PFRAME_RESUME}"
elif [[ -n "${INIT_CKPT}" ]]; then
  [[ -f "${INIT_CKPT}" ]] || {
    echo "Missing I-frame ckpt: ${INIT_CKPT}"
    exit 1
  }
  ckpt_args=(--checkpoint "${INIT_CKPT}")
  echo "I-frame ckpt (ref only): ${INIT_CKPT}"
else
  echo "I-frame ckpt: (none — ref_encoder trains from scratch)"
fi

p_codec_args=()
if [[ -z "${PFRAME_RESUME}" && -n "${P_CODEC_INIT}" ]]; then
  [[ -f "${P_CODEC_INIT}" ]] || {
    echo "Missing P_CODEC_INIT: ${P_CODEC_INIT}"
    exit 1
  }
  p_codec_args=(--p-codec-init "${P_CODEC_INIT}")
  echo "P-codec init:            ${P_CODEC_INIT}"
else
  echo "P-codec init:            (none, from scratch)"
fi

echo "Stage:      pframe (ref I-frame feat + decoder fuse -> canny)"
echo "Model:      HPCM_Video_PFrame"
echo "Lambda:     ${LAMBDA}"
echo "Save every: ${SAVE_INTERVAL} epochs"
echo "Data:       ${DATA_ROOT}"
echo "Save:       ${SAVE_PATH}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" train_video.py \
  --stage pframe \
  --model_name HPCM_Video_PFrame \
  --dataset-root "${DATA_ROOT}" \
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
  2>&1 | tee "${OUT_ROOT}/train.log"
