#!/usr/bin/env bash
# Train HPCM_Canny1ch_Spconv: spconv sparse conv in g_a/g_s/h_a/h_s (ablation).
source "$(dirname "$0")/env.sh"
set -euo pipefail

LAMBDA="${LAMBDA:-0.00011}"
PATCH_SIZE="${PATCH_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"

CANNY_TRAIN="${CANNY_TRAIN:-/data/Dataset/RealVSR_GT_test_iframe_all/canny}"
CANNY_TEST="${CANNY_TEST:-/data/Dataset/RealVSR_GT_test_iframe_all/canny}"
INIT_CKPT="${INIT_CKPT:-}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/train_canny1ch_spconv_lambda${LAMBDA}}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"
TRAIN_LOG="${TRAIN_LOG:-${OUT_ROOT}/train.log}"

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

"${PYTHON}" -c "import spconv; print('spconv', spconv.__version__)"

[[ -d "${CANNY_TRAIN}" ]] || { echo "Missing train canny dir: ${CANNY_TRAIN}"; exit 1; }
[[ -d "${CANNY_TEST}" ]] || { echo "Missing test canny dir: ${CANNY_TEST}"; exit 1; }

ckpt_args=()
if [[ -n "${INIT_CKPT}" ]]; then
  [[ -f "${INIT_CKPT}" ]] || { echo "Missing init ckpt: ${INIT_CKPT}"; exit 1; }
  ckpt_args=(--checkpoint "${INIT_CKPT}")
fi

echo "Model:      HPCM_Canny1ch_Spconv"
echo "Lambda:     ${LAMBDA}"
echo "Train:      ${CANNY_TRAIN}"
echo "Save:       ${SAVE_PATH}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" train.py \
  --model_name HPCM_Canny1ch_Spconv \
  --train_dataset "${CANNY_TRAIN}" \
  --test_dataset "${CANNY_TEST}" \
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
  2>&1 | tee -a "${TRAIN_LOG}"
