#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Train HPCM_Canny1ch as a 1-channel canny codec (lambda=0.00105, canny 512x512 input).
#
# Usage:
#   bash scripts/train_hpcm_canny.sh
#   PREPARE_CANNY=1 bash scripts/train_hpcm_canny.sh
#   INIT_CKPT=checkpoints/HPCM_Base/MSE/0.0018.pth.tar bash scripts/train_hpcm_canny.sh  # finetune

set -euo pipefail

# LIC_ROOT from env.sh


PREPARE_CANNY="${PREPARE_CANNY:-0}"
MAX_CANNY_IMAGES="${MAX_CANNY_IMAGES:-0}"
HR_ROOT="${HR_ROOT:-/data/Dataset/df2k_ost/GT}"
CANNY_TRAIN="${CANNY_TRAIN:-/data/Dataset/HPCM_canny_train}"
CANNY_TEST="${CANNY_TEST:-/data/Dataset/DIV2K/DIV2K_valid_100_512_128/canny}"

# Empty = train from scratch. Set to a .pth.tar path to finetune.
INIT_CKPT="${INIT_CKPT:-}"
LAMBDA="${LAMBDA:-0.00105}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3001}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATCH_SIZE="${PATCH_SIZE:-256}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/train_canny1ch_lambda0.00105}"
SAVE_PATH="${OUT_ROOT}/checkpoints"
LOG_DIR="${OUT_ROOT}/logs"
TRAIN_LOG="${TRAIN_LOG:-${OUT_ROOT}/train.log}"

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

if [[ "${PREPARE_CANNY}" == "1" ]]; then
  extra=()
  if [[ "${MAX_CANNY_IMAGES}" != "0" ]]; then
    extra+=(--max-images "${MAX_CANNY_IMAGES}")
  fi
  "${PYTHON}" "${CODEC_ROOT}/scripts/hpcm/prepare_canny_dataset.py" \
    --hr-root "${HR_ROOT}" \
    --out-dir "${CANNY_TRAIN}" \
    --size 512 \
    "${extra[@]}"
fi

[[ -d "${CANNY_TRAIN}" ]] || { echo "Missing train canny dir: ${CANNY_TRAIN}"; exit 1; }
[[ -d "${CANNY_TEST}" ]] || { echo "Missing test canny dir: ${CANNY_TEST}"; exit 1; }

ckpt_args=()
if [[ -n "${INIT_CKPT}" ]]; then
  [[ -f "${INIT_CKPT}" ]] || { echo "Missing init ckpt: ${INIT_CKPT}"; exit 1; }
  ckpt_args=(--checkpoint "${INIT_CKPT}")
  echo "Init ckpt:  ${INIT_CKPT}"
else
  echo "Init ckpt:  (none, from scratch)"
fi

echo "Lambda:     ${LAMBDA}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train:      ${CANNY_TRAIN}"
echo "Test:       ${CANNY_TEST}"
echo "Save:       ${SAVE_PATH}"
echo "Stdout log: ${TRAIN_LOG}"
echo "TB log:     ${LOG_DIR}/HPCM_Canny1ch_lmbda${LAMBDA}/  (tensorboard --logdir ${LOG_DIR})"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" train.py \
  --model_name HPCM_Canny1ch \
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
  --clip_max_norm 1.0 \
  2>&1 | tee -a "${TRAIN_LOG}"
