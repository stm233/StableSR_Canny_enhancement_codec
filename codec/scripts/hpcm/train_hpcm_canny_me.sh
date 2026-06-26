#!/usr/bin/env bash
# Train HPCM_Canny1ch_ME: MinkowskiEngine conv in g_a/g_s/h_a/h_s (ablation).
source "$(dirname "$0")/env.sh"
set -euo pipefail

LAMBDA="${LAMBDA:-0.00011}"
PATCH_SIZE="${PATCH_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-300}"

TRAIN_DATASET="${TRAIN_DATASET:-/data/Dataset/RealVSR_GT_test_iframe_all/canny}"
TEST_DATASET="${TEST_DATASET:-/data/Dataset/RealVSR_GT_test_iframe_all/canny}"

LOG_DIR="${LOG_DIR:-/data/Dataset/LIC-HPCM_outputs/train_canny1ch_me_lambda${LAMBDA}}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/checkpoints}"

cd "${CODEC_ROOT}"
"${PYTHON}" train.py \
  --model_name HPCM_Canny1ch_ME \
  --lambda "${LAMBDA}" \
  --patch_size "${PATCH_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --train-dataset "${TRAIN_DATASET}" \
  --test-dataset "${TEST_DATASET}" \
  --log_dir "${LOG_DIR}" \
  --save_path "${SAVE_PATH}" \
  --cuda
