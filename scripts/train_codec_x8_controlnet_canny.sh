#!/usr/bin/env bash
# Fine-tune StableSR+ControlNet on lossy codec LQ (64) + lossy GOP Canny, x8 SR.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"

CFG="configs/stableSRNew/v2-finetune_text_T_512_controlnet_canny_codec_x8.yaml"
NAME="${NAME:-stablesr_cn_canny_codec_x8}"
GPUS="${GPUS:-0,}"

echo "Config: ${CFG}"
echo "Name:   ${NAME}"
echo "GPUs:   ${GPUS}"
echo "Init:   epoch=000012.ckpt (see yaml model.params.ckpt_path)"
echo ""

"${PYTHON}" main.py --train \
  --base "${CFG}" \
  --gpus "${GPUS}" \
  --name "${NAME}" \
  --scale_lr False \
  2>&1 | tee "logs/train_${NAME}.log"
