#!/usr/bin/env bash
# Fine-tune StableSR+ControlNet on HQ-VSR_SR_codec:
#   LQ: DCVC lossy lq64_lossy | Canny: lossless canny256 | GT: hq512
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"

CODEC_ROOT="${CODEC_ROOT:-/data/Dataset/HQ-VSR_SR_codec}"
PREPARE_MANIFEST="${PREPARE_MANIFEST:-1}"
CFG="${CFG:-configs/stableSRNew/v2-finetune_text_T_512_controlnet_canny_hqvsr_codec_x8.yaml}"
NAME="${NAME:-stablesr_cn_hqvsr_codec_x8}"
GPUS="${GPUS:-0,}"

if [[ "${PREPARE_MANIFEST}" == "1" ]]; then
  echo "[1/2] Building StableSR CN training manifest..."
  "${PYTHON}" scripts/prepare_hqvsr_stablesr_cn_manifest.py --codec-root "${CODEC_ROOT}"
fi

echo "[2/2] Training StableSR + ControlNet"
echo "  Config: ${CFG}"
echo "  Name:   ${NAME}"
echo "  GPUs:   ${GPUS}"
echo ""

"${PYTHON}" main.py --train \
  --base "${CFG}" \
  --gpus "${GPUS}" \
  --name "${NAME}" \
  --scale_lr False \
  2>&1 | tee "logs/train_${NAME}.log"
