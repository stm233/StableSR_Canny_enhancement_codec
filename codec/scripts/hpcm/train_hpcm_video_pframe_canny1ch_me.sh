#!/usr/bin/env bash
# P-frame on HPCM_Canny1ch_ME (frozen lossy I-frame ref + multi-scale fusion).
source "$(dirname "$0")/env.sh"
set -euo pipefail

LAMBDA="${LAMBDA:-11}"
INIT_CKPT="${INIT_CKPT:-/data/Dataset/LIC-HPCM_outputs/train_canny1ch_me_lambda0.00011/checkpoints/HPCM_Canny1ch_ME_lmbda0.00011/epoch_best.pth.tar}"
DATA_ROOT="${DATA_ROOT:-/data/Dataset/RealVSR_GT_test_iframe_all}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-50}"

LOG_DIR="${LOG_DIR:-/data/Dataset/LIC-HPCM_outputs/train_pframe_canny1ch_me_lambda${LAMBDA}}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/checkpoints}"

cd "${CODEC_ROOT}"
"${PYTHON}" train_video.py \
  --stage pframe \
  --model_name HPCM_Video_PFrame_Canny1ch_ME \
  --lmbda "${LAMBDA}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${INIT_CKPT}" \
  --log_dir "${LOG_DIR}" \
  --save_path "${SAVE_PATH}" \
  --cuda
