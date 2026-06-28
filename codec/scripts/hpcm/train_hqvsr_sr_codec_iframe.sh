#!/usr/bin/env bash
# I-frame: compress canny256 with cond=canny64_lossy (partial spconv g_a/g_s + multi-scale fusion).
source "$(dirname "$0")/env.sh"
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_SR_codec}"
PREPARE_MANIFEST="${PREPARE_MANIFEST:-1}"
LAMBDA="${LAMBDA:-0.00105}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-3001}"
PATCH_SIZE="${PATCH_SIZE:-256}"
VAL_SAMPLES="${VAL_SAMPLES:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/hqvsr_sr_codec_iframe_cond}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

if [[ "${PREPARE_MANIFEST}" == "1" ]]; then
  "${PYTHON}" "$(dirname "$0")/prepare_hqvsr_sr_codec_manifest.py" --codec-root "${DATA_ROOT}"
fi

cd "${LIC_ROOT}"
"${PYTHON}" train_video.py \
  --stage iframe \
  --hqvsr-codec \
  --model_name HPCM_Canny1ch_Spconv_Cond \
  --dataset-root "${DATA_ROOT}" \
  --val-samples "${VAL_SAMPLES}" \
  --lambda "${LAMBDA}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --patch-size "${PATCH_SIZE}" "${PATCH_SIZE}" \
  --save_path "${SAVE_PATH}" \
  --log_dir "${LOG_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --clip_max_norm 1.0 \
  2>&1 | tee "${OUT_ROOT}/train.log"
