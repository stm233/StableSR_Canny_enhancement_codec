#!/usr/bin/env bash
# SR x8 with uncompressed LR + HPCM-compressed canny hints.
# METRIC=mse     -> 6 MSE points × 2 SR models = 12 runs (default)
# METRIC=msssim  -> 6 MS-SSIM points × 2 = 12 runs
# METRIC=both    -> 12 rates × 2 = 24 runs
#
#   LQ (init-img):  /data/Dataset/DIV2K/DIV2K_valid_100_512_64/LR_64  (uncompressed)
#   Canny hint:     /data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/canny/{MSE,MSSSIM}/lambda_*/images
#   GT (metrics):   /data/Dataset/DIV2K/DIV2K_valid_100_512_128/HR_512
#
# Usage:
#   bash scripts/run_canny_codec_sr_x8_rd.sh
#   RUN_INFER=0 bash scripts/run_canny_codec_sr_x8_rd.sh

set -euo pipefail

STABLESR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

LQ_DIR="/data/Dataset/DIV2K/DIV2K_valid_100_512_64/LR_64"
GT_DIR="/data/Dataset/DIV2K/DIV2K_valid_100_512_128/HR_512"
CANNY_HPCM_ROOT="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/canny"

OUT_ROOT="/data/Dataset/StableSR-TestSets/codec_rd/canny_codec_x8"
LOG_DIR="${OUT_ROOT}/logs"
METRICS_DIR="${OUT_ROOT}/metrics"

MSE_LAMBDAS=(0.0018 0.0035 0.0067 0.013 0.025 0.0483)
MSSSIM_LAMBDAS=(2.4 4.58 8.73 16.64 31.73 60.5)
# mse | msssim | both  (INCLUDE_MSSSIM=1 is alias for METRIC=both)
if [[ "${INCLUDE_MSSSIM:-0}" == "1" ]]; then
  METRIC="both"
else
  METRIC="${METRIC:-mse}"
fi

BASELINE_CFG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
BASELINE_CKPT="checkpoints/stablesr_000117.ckpt"
VQGAN_CKPT="checkpoints/vqgan_cfw_00011.ckpt"

CN_CFG="configs/stableSRNew/v2-finetune_text_T_512_controlnet_canny.yaml"
CN_CKPT="/home/exx/Documents/Tianma/StableSR/logs/2026-06-06T00-08-37_stablesr_cn_canny/checkpoints/epoch=000003.ckpt"
CN_VQGAN_CFG="configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml"

DDPM_STEPS=100
INPUT_SIZE=512
DEC_W=0.5
SEED=42

mkdir -p "${LOG_DIR}" "${METRICS_DIR}" \
  "${OUT_ROOT}/baseline" "${OUT_ROOT}/controlnet_canny_e3"

echo "LQ (uncompressed): ${LQ_DIR}"
echo "Canny HPCM root:   ${CANNY_HPCM_ROOT}"
echo "OUT_ROOT:          ${OUT_ROOT}"
echo ""

run_baseline() {
  local metric="$1"
  local lam="$2"
  local tag="${metric}_${lam}"
  local out_dir="${OUT_ROOT}/baseline/${tag}_x8"
  local log="${LOG_DIR}/baseline_${tag}_x8.log"
  local metrics_json="${METRICS_DIR}/baseline_${tag}_x8.json"

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -d "${out_dir}" ]] && \
       [[ "$(find "${out_dir}" -maxdepth 1 -name '*.png' ! -name '*_canny.png' | wc -l)" -gt 0 ]]; then
      echo "[SKIP infer] baseline ${tag}"
    else
      echo "======== baseline | canny ${tag} | LQ=raw LR_64 | x8 ========"
      mkdir -p "${out_dir}"
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/sr_val_ddpm_text_T_vqganfin_old.py \
        --config "${BASELINE_CFG}" \
        --ckpt "${BASELINE_CKPT}" \
        --vqgan_ckpt "${VQGAN_CKPT}" \
        --init-img "${LQ_DIR}" \
        --outdir "${out_dir}" \
        --input_size "${INPUT_SIZE}" \
        --ddpm_steps "${DDPM_STEPS}" \
        --dec_w "${DEC_W}" \
        --colorfix_type adain \
        --n_samples 1 \
        --seed "${SEED}" \
        2>&1 | tee "${log}"
    fi
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics_json}" ]]; then
      echo "[SKIP eval] ${metrics_json}"
    else
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/eval_sr_metrics.py \
        --gt-dir "${GT_DIR}" \
        --out-dir "${out_dir}" \
        --device "${DEVICE}" \
        --skip-fid \
        --json-out "${metrics_json}"
    fi
  fi
}

run_controlnet() {
  local metric="$1"
  local lam="$2"
  local tag="${metric}_${lam}"
  local canny_dir="${CANNY_HPCM_ROOT}/${metric}/lambda_${lam}/images"
  local out_dir="${OUT_ROOT}/controlnet_canny_e3/${tag}_x8"
  local log="${LOG_DIR}/controlnet_${tag}_x8.log"
  local metrics_json="${METRICS_DIR}/controlnet_canny_e3_${tag}_x8.json"

  [[ -d "${canny_dir}" ]] || { echo "[SKIP] missing canny: ${canny_dir}"; return 1; }

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -d "${out_dir}" ]] && \
       [[ "$(find "${out_dir}" -maxdepth 1 -name '*.png' ! -name '*_canny.png' | wc -l)" -gt 0 ]]; then
      echo "[SKIP infer] controlnet ${tag}"
    else
      echo "======== controlnet | canny ${tag} | LQ=raw LR_64 | x8 ========"
      mkdir -p "${out_dir}"
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/sr_val_ddpm_text_T_vqganfin_hqCanny.py \
        --config "${CN_CFG}" \
        --ckpt "${CN_CKPT}" \
        --vqgan-config "${CN_VQGAN_CFG}" \
        --vqgan_ckpt "${VQGAN_CKPT}" \
        --init-img "${LQ_DIR}" \
        --gt-img "${GT_DIR}" \
        --canny-dir "${canny_dir}" \
        --outdir "${out_dir}" \
        --input_size "${INPUT_SIZE}" \
        --ddpm_steps "${DDPM_STEPS}" \
        --dec_w "${DEC_W}" \
        --colorfix_type adain \
        --n_samples 1 \
        --seed "${SEED}" \
        --save_canny_vis \
        2>&1 | tee "${log}"
    fi
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics_json}" ]]; then
      echo "[SKIP eval] ${metrics_json}"
    else
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/eval_sr_metrics.py \
        --gt-dir "${GT_DIR}" \
        --out-dir "${out_dir}" \
        --device "${DEVICE}" \
        --skip-fid \
        --json-out "${metrics_json}"
    fi
  fi
}

if [[ "${METRIC}" == "mse" || "${METRIC}" == "both" ]]; then
  for lam in "${MSE_LAMBDAS[@]}"; do
    run_baseline "MSE" "${lam}"
    run_controlnet "MSE" "${lam}"
  done
fi

if [[ "${METRIC}" == "msssim" || "${METRIC}" == "both" ]]; then
  for lam in "${MSSSIM_LAMBDAS[@]}"; do
    run_baseline "MSSSIM" "${lam}"
    run_controlnet "MSSSIM" "${lam}"
  done
fi

if [[ "${RUN_PLOT}" == "1" ]]; then
  cd "${STABLESR_ROOT}"
  "${PYTHON}" scripts/plot_canny_codec_sr_rd_curves.py \
    --metrics-dir "${METRICS_DIR}" \
    --canny-hpcm-root "${CANNY_HPCM_ROOT}" \
    --out-dir "${OUT_ROOT}/rd_curves" \
    --metric "${METRIC}"
fi

echo ""
case "${METRIC}" in
  mse)     echo "Done. 12 runs (6 MSE × 2 SR models)" ;;
  msssim)  echo "Done. 12 runs (6 MS-SSIM × 2 SR models)" ;;
  both)    echo "Done. 24 runs (12 rates × 2 SR models)" ;;
esac
echo "Outputs: ${OUT_ROOT}/"
echo "Curves:  ${OUT_ROOT}/rd_curves/"
