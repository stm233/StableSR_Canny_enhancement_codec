#!/usr/bin/env bash
# Combined-bitrate SR experiment (7 runs):
#   1) StableSR baseline: HPCM LQ @ highest MSE (lambda=0.0483) only
#   6) ControlNet: same LQ + canny @ MS-SSIM 6 rate points
#
# Total BPP @ 512:
#   baseline  = LQ_bpp@64 / 64
#   controlnet = LQ_bpp@64 / 64 + canny_bpp@512
#
# Usage:
#   bash scripts/run_hq_lq_canny_msssim_combined_rd.sh
#   RUN_INFER=0 bash scripts/run_hq_lq_canny_msssim_combined_rd.sh

set -euo pipefail

STABLESR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"
export PYTHONPATH="${STABLESR_ROOT}:/home/exx/Documents/Tianma/src/taming-transformers:${PYTHONPATH:-}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

LQ_DIR="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/lr64/MSE/lambda_0.0483/images"
LQ_HPCM_SUMMARY="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/lr64/MSE/lambda_0.0483"
CANNY_HPCM_ROOT="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/canny/MSSSIM"
GT_DIR="/data/Dataset/DIV2K/DIV2K_valid_100_512_128/HR_512"

OUT_ROOT="/data/Dataset/StableSR-TestSets/codec_rd/hq_lq_canny_msssim_combined"
LOG_DIR="${OUT_ROOT}/logs"
METRICS_DIR="${OUT_ROOT}/metrics"

MSSSIM_LAMBDAS=(2.4 4.58 8.73 16.64 31.73 60.5)
LQ_BPP_SCALE=64

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

echo "LQ (HPCM MSE 0.0483): ${LQ_DIR}"
echo "Canny (MS-SSIM):      ${CANNY_HPCM_ROOT}"
echo "OUT_ROOT:             ${OUT_ROOT}"
echo ""

run_baseline() {
  local out_dir="${OUT_ROOT}/baseline/LQ_MSE_0.0483_x8"
  local log="${LOG_DIR}/baseline_LQ_MSE_0.0483_x8.log"
  local metrics_json="${METRICS_DIR}/baseline_LQ_MSE_0.0483_x8.json"

  [[ -d "${LQ_DIR}" ]] || { echo "[SKIP] missing LQ: ${LQ_DIR}"; return 1; }

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -d "${out_dir}" ]] && \
       [[ "$(find "${out_dir}" -maxdepth 1 -name '*.png' ! -name '*_canny.png' | wc -l)" -gt 0 ]]; then
      echo "[SKIP infer] baseline LQ_MSE_0.0483"
    else
      echo "======== baseline | LQ=MSE_0.0483 (hq) | x8 ========"
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
  local lam="$1"
  local tag="MSSSIM_${lam}"
  local canny_dir="${CANNY_HPCM_ROOT}/lambda_${lam}/images"
  local out_dir="${OUT_ROOT}/controlnet_canny_e3/${tag}_x8"
  local log="${LOG_DIR}/controlnet_${tag}_x8.log"
  local metrics_json="${METRICS_DIR}/controlnet_canny_e3_${tag}_x8.json"

  [[ -d "${canny_dir}" ]] || { echo "[SKIP] missing canny: ${canny_dir}"; return 1; }

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -d "${out_dir}" ]] && \
       [[ "$(find "${out_dir}" -maxdepth 1 -name '*.png' ! -name '*_canny.png' | wc -l)" -gt 0 ]]; then
      echo "[SKIP infer] controlnet ${tag}"
    else
      echo "======== controlnet | LQ=MSE_0.0483 + canny ${tag} | x8 ========"
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

run_baseline

for lam in "${MSSSIM_LAMBDAS[@]}"; do
  run_controlnet "${lam}"
done

if [[ "${RUN_PLOT}" == "1" ]]; then
  cd "${STABLESR_ROOT}"
  "${PYTHON}" scripts/plot_hq_lq_canny_msssim_combined_rd_curves.py \
    --metrics-dir "${METRICS_DIR}" \
    --lq-hpcm-summary "${LQ_HPCM_SUMMARY}" \
    --canny-hpcm-root "${CANNY_HPCM_ROOT}" \
    --out-dir "${OUT_ROOT}/rd_curves" \
    --lq-bpp-scale "${LQ_BPP_SCALE}"
fi

echo ""
echo "Done. 7 runs (1 baseline + 6 controlnet)"
echo "Curves: ${OUT_ROOT}/rd_curves/"
