#!/usr/bin/env bash
# RealVSR: GOP codec (Canny1ch I+P) -> StableSR ControlNet Canny x8 SR.
#
# StableSR expects paired folders with SAME filenames:
#   --init-img  : HPCM_Base decompressed LR64 RGB  (016_00000.png ...)
#   --canny-dir : GOP decompressed Canny           (016_00000.png ...)
#   --gt-img    : GT 512 for metrics               (016_00000.png ...)
#
# Prereq:
#   1) bash codec/scripts/hpcm/prepare_realvsr_lr64_flat.sh
#   2) LAMBDA=0.0035 bash codec/scripts/hpcm/test_realvsr_hpcm_base.sh  (LQ init-img)
#   3) GOP infer (this script step 1)
#
# Usage:
#   RUN_CODEC=1 RUN_SR=1 bash codec/scripts/hpcm/run_realvsr_codec_gop_to_stablesr.sh
#   RUN_CODEC=0 RUN_SR=1 LQ_LAMBDA=0.0035 bash ...   # SR only (canny already exported)

source "$(dirname "$0")/env.sh"
set -euo pipefail

STABLESR_ROOT="${STABLESR_ROOT:-/home/exx/Documents/Tianma/StableSR}"

PFRAME_CKPT="${PFRAME_CKPT:-${CODEC_ROOT}/checkpoints/P_frame/lambda_11_epoch_7.tar}"
DATA_ROOT="${DATA_ROOT:-/data/Dataset/RealVSR_GT_test_iframe_all}"
GT_DIR="${GT_DIR:-/data/Dataset/RealVSR_GT_test_iframe_all_flat/gt512}"
LQ_LAMBDA="${LQ_LAMBDA:-0.0035}"
LQ_ROOT="${LQ_ROOT:-/data/Dataset/LIC-HPCM_outputs/RealVSR_GT_test_lr64/MSE/lambda_${LQ_LAMBDA}/images}"

NUM_P="${NUM_P:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_GOPS_PER_VIDEO="${MAX_GOPS_PER_VIDEO:-0}"
DEVICE="${DEVICE:-cuda}"

CODEC_OUT="${CODEC_OUT:-/data/Dataset/LIC-HPCM_outputs/RealVSR_gop_canny1ch_l${LQ_LAMBDA}}"
CANNY_DIR="${CODEC_OUT}/canny/images"
SR_OUT="${SR_OUT:-/data/Dataset/StableSR-TestSets/RealVSR_gop_canny1ch_l${LQ_LAMBDA}_x8}"

CN_CFG="configs/stableSRNew/v2-finetune_text_T_512_controlnet_canny.yaml"
CN_CKPT="${CN_CKPT:-/home/exx/Documents/Tianma/StableSR/logs/2026-06-06T00-08-37_stablesr_cn_canny/checkpoints/epoch=000012.ckpt}"
VQGAN_CKPT="${VQGAN_CKPT:-checkpoints/vqgan_cfw_00011.ckpt}"

RUN_CODEC="${RUN_CODEC:-1}"
RUN_SR="${RUN_SR:-1}"

if [[ "${RUN_CODEC}" == "1" ]]; then
  PFRAME_CKPT="${PFRAME_CKPT}" \
  DATA_ROOT="${DATA_ROOT}" \
  NUM_P="${NUM_P}" \
  MAX_VIDEOS="${MAX_VIDEOS}" \
  MAX_GOPS_PER_VIDEO="${MAX_GOPS_PER_VIDEO}" \
  OUT_ROOT="${CODEC_OUT}" \
  DEVICE="${DEVICE}" \
  bash "${CODEC_ROOT}/scripts/hpcm/infer_realvsr_gop_canny1ch.sh"
fi

[[ -d "${CANNY_DIR}" ]] || { echo "Missing canny export: ${CANNY_DIR}"; exit 1; }
[[ -d "${LQ_ROOT}" ]] || {
  echo "Missing LQ for --init-img: ${LQ_ROOT}"
  echo "Run: LAMBDA=${LQ_LAMBDA} bash codec/scripts/hpcm/test_realvsr_hpcm_base.sh"
  exit 1
}
[[ -d "${GT_DIR}" ]] || { echo "Missing GT: ${GT_DIR}"; exit 1; }

echo ""
echo "StableSR pairing (same filenames, e.g. 016_00000.png):"
echo "  --init-img  ${LQ_ROOT}"
echo "  --canny-dir ${CANNY_DIR}"
echo "  --gt-img    ${GT_DIR}"
echo ""

if [[ "${RUN_SR}" == "1" ]]; then
  mkdir -p "${SR_OUT}"
  cd "${STABLESR_ROOT}"
  "${PYTHON}" scripts/sr_val_ddpm_text_T_vqganfin_hqCanny.py \
    --config "${CN_CFG}" \
    --ckpt "${CN_CKPT}" \
    --vqgan-config configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml \
    --vqgan_ckpt "${VQGAN_CKPT}" \
    --init-img "${LQ_ROOT}" \
    --canny-dir "${CANNY_DIR}" \
    --gt-img "${GT_DIR}" \
    --outdir "${SR_OUT}" \
    --input_size 512 \
    --ddpm_steps 100 \
    --dec_w 0.5 \
    --colorfix_type adain \
    --n_samples 1 \
    --compute_metrics \
    --save_canny_vis \
    2>&1 | tee "${SR_OUT}/infer.log"
  echo "SR outputs: ${SR_OUT}"
fi
