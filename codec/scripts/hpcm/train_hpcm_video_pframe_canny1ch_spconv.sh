#!/usr/bin/env bash
# P-frame on HPCM_Canny1ch_Spconv (frozen lossy I-frame ref + multi-scale fusion).
source "$(dirname "$0")/env.sh"
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset/RealVSR_GT_test_iframe_all}"
INIT_CKPT="${INIT_CKPT:-/data/Dataset/LIC-HPCM_outputs/train_canny1ch_spconv_lambda0.00011/checkpoints/HPCM_Canny1ch_Spconv_lmbda0.00011/epoch_best.pth.tar}"
LAMBDA="${LAMBDA:-0.00011}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-300}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATCH_SIZE="${PATCH_SIZE:-256}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/video_pframe_canny1ch_spconv_lambda${LAMBDA}}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"
[[ -f "${DATA_ROOT}/manifest_pframe.jsonl" ]] || { echo "Missing manifest_pframe.jsonl"; exit 1; }

ckpt_args=()
[[ -n "${INIT_CKPT}" && -f "${INIT_CKPT}" ]] && ckpt_args=(--checkpoint "${INIT_CKPT}")

cd "${CODEC_ROOT}"
"${PYTHON}" train_video.py \
  --stage pframe \
  --model_name HPCM_Video_PFrame_Canny1ch_Spconv \
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
