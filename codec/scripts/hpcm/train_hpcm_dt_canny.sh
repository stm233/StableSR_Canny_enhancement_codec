#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Train HPCM_DT1ch: DT 3ch in (R,G,B), decoder outputs Canny; MSE on Canny target.
#
# Usage:
#   LAMBDA=0.00105 bash codec/scripts/hpcm/train_hpcm_dt_canny.sh
#   PREPARE_DT=1 LAMBDA=0.00105 bash codec/scripts/hpcm/train_hpcm_dt_canny.sh

set -euo pipefail

PREPARE_DT="${PREPARE_DT:-0}"
CANNY_TRAIN="${CANNY_TRAIN:-/data/Dataset/HPCM_canny_train}"
CANNY_TEST="${CANNY_TEST:-/data/Dataset/DIV2K/DIV2K_valid_100_512_128/canny}"
DT_TRAIN="${DT_TRAIN:-/data/Dataset/HPCM_dt_canny_train}"
DT_TEST="${DT_TEST:-/data/Dataset/HPCM_dt_canny_test}"

INIT_CKPT="${INIT_CKPT:-}"
LAMBDA="${LAMBDA:-0.00105}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3001}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATCH_SIZE="${PATCH_SIZE:-256}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/train_dt1ch_lambda0.00105}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"
TRAIN_LOG="${TRAIN_LOG:-${OUT_ROOT}/train.log}"

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

if [[ "${PREPARE_DT}" == "1" ]]; then
  "${PYTHON}" "${CODEC_ROOT}/scripts/hpcm/prepare_dt_canny_dataset.py" \
    --canny-dir "${CANNY_TRAIN}" --out-dir "${DT_TRAIN}"
  "${PYTHON}" "${CODEC_ROOT}/scripts/hpcm/prepare_dt_canny_dataset.py" \
    --canny-dir "${CANNY_TEST}" --out-dir "${DT_TEST}"
fi

# DT is computed on-the-fly if cache dirs missing; use CANNY_* as fallback
TRAIN_DATA="${CANNY_TRAIN}"
TEST_DATA="${CANNY_TEST}"
DT_SOURCE="canny_l"
if [[ -d "${DT_TRAIN}" ]]; then TRAIN_DATA="${DT_TRAIN}"; DT_SOURCE="dt_rgb"; fi
if [[ -d "${DT_TEST}" ]]; then TEST_DATA="${DT_TEST}"; fi

[[ -d "${TRAIN_DATA}" ]] || { echo "Missing train dir: ${TRAIN_DATA}"; exit 1; }
[[ -d "${TEST_DATA}" ]] || { echo "Missing test dir: ${TEST_DATA}"; exit 1; }

ckpt_args=()
if [[ -n "${INIT_CKPT}" ]]; then
  [[ -f "${INIT_CKPT}" ]] || { echo "Missing init ckpt: ${INIT_CKPT}"; exit 1; }
  ckpt_args=(--checkpoint "${INIT_CKPT}")
fi

echo "Model:      HPCM_DT1ch"
echo "Lambda:     ${LAMBDA}"
echo "Train:      ${TRAIN_DATA}"
echo "Test:       ${TEST_DATA}"
echo "DT source:  ${DT_SOURCE}"
echo "Save:       ${SAVE_PATH}"
echo "Stdout log: ${TRAIN_LOG}"
echo "TB log:     ${LOG_DIR}/HPCM_DT1ch_lmbda${LAMBDA}/"
echo ""

cd "${CODEC_ROOT}"
canny_args=()
if [[ "${DT_SOURCE}" == "dt_rgb" ]]; then
  canny_args=(--canny-dataset "${CANNY_TRAIN}" --canny-test-dataset "${CANNY_TEST}")
fi

"${PYTHON}" train.py \
  --model_name HPCM_DT1ch \
  --dt-source "${DT_SOURCE}" \
  --train_dataset "${TRAIN_DATA}" \
  --test_dataset "${TEST_DATA}" \
  "${canny_args[@]}" \
  "${ckpt_args[@]}" \
  --lambda "${LAMBDA}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --patch-size "${PATCH_SIZE}" "${PATCH_SIZE}" \
  --save_path "${SAVE_PATH}" \
  --log_dir "${LOG_DIR}" \
  --clip_max_norm 1.0 \
  2>&1 | tee -a "${TRAIN_LOG}"
