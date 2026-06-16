#!/usr/bin/env bash
# HPCM decompressed LQ (64x64) -> StableSR x8 (512) R-D experiment.
# For each of 6 MSE lambdas (in order), run:
#   1) StableSR baseline
#   2) ControlNet Canny SR (canny from uncompressed GT, not compressed canny)
# Then evaluate metrics and plot PSNR(Y)/MS-SSIM(Y)/LPIPS vs bpp@512.
#
# Usage:
#   bash scripts/run_hpcm_lq_sr_x8_rd.sh
#   RUN_INFER=0 bash scripts/run_hpcm_lq_sr_x8_rd.sh   # only eval + plot
#   SKIP_EXISTING=0 bash scripts/run_hpcm_lq_sr_x8_rd.sh

set -euo pipefail

STABLESR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

HPCM_ROOT="/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/lr64/MSE"
GT_DIR="/data/Dataset/DIV2K/DIV2K_valid_100_512_128/HR_512"
OUT_ROOT="/data/Dataset/StableSR-TestSets/codec_rd/x8"
LOG_DIR="${OUT_ROOT}/logs"
METRICS_DIR="${OUT_ROOT}/metrics"

# Order matters: low -> high bpp
LAMBDAS=(0.0018 0.0035 0.0067 0.013 0.025 0.0483)

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
SCALE_BPP=64   # bpp@512 = bpp@64 / SCALE_BPP

mkdir -p "${LOG_DIR}" "${METRICS_DIR}" \
  "${OUT_ROOT}/baseline" "${OUT_ROOT}/controlnet_canny_e3"

echo "STABLESR_ROOT=${STABLESR_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "DEVICE=${DEVICE}"
echo ""

run_baseline() {
  local lam="$1"
  local lq_dir="${HPCM_ROOT}/lambda_${lam}/images"
  local out_dir="${OUT_ROOT}/baseline/MSE_${lam}_x8"
  local log="${LOG_DIR}/baseline_MSE_${lam}_x8.log"
  local metrics_json="${METRICS_DIR}/baseline_MSE_${lam}_x8.json"

  [[ -d "${lq_dir}" ]] || { echo "[SKIP] missing LQ: ${lq_dir}"; return 1; }

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -d "${out_dir}" ]] && \
       [[ "$(find "${out_dir}" -maxdepth 1 -name '*.png' | wc -l)" -gt 0 ]]; then
      echo "[SKIP infer] baseline MSE_${lam}"
    else
      echo "======== baseline | MSE lambda=${lam} | x8 ========"
      mkdir -p "${out_dir}"
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/sr_val_ddpm_text_T_vqganfin_old.py \
        --config "${BASELINE_CFG}" \
        --ckpt "${BASELINE_CKPT}" \
        --vqgan_ckpt "${VQGAN_CKPT}" \
        --init-img "${lq_dir}" \
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
  local lq_dir="${HPCM_ROOT}/lambda_${lam}/images"
  local out_dir="${OUT_ROOT}/controlnet_canny_e3/MSE_${lam}_x8"
  local log="${LOG_DIR}/controlnet_MSE_${lam}_x8.log"
  local metrics_json="${METRICS_DIR}/controlnet_canny_e3_MSE_${lam}_x8.json"

  [[ -d "${lq_dir}" ]] || { echo "[SKIP] missing LQ: ${lq_dir}"; return 1; }

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -d "${out_dir}" ]] && \
       [[ "$(find "${out_dir}" -maxdepth 1 -name '*.png' | wc -l)" -gt 0 ]]; then
      echo "[SKIP infer] controlnet MSE_${lam}"
    else
      echo "======== controlnet_canny | MSE lambda=${lam} | x8 ========"
      mkdir -p "${out_dir}"
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/sr_val_ddpm_text_T_vqganfin_hqCanny.py \
        --config "${CN_CFG}" \
        --ckpt "${CN_CKPT}" \
        --vqgan-config "${CN_VQGAN_CFG}" \
        --vqgan_ckpt "${VQGAN_CKPT}" \
        --init-img "${lq_dir}" \
        --gt-img "${GT_DIR}" \
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

for lam in "${LAMBDAS[@]}"; do
  run_baseline "${lam}"
  run_controlnet "${lam}"
done

if [[ "${RUN_PLOT}" == "1" ]]; then
  cd "${STABLESR_ROOT}"
  "${PYTHON}" scripts/plot_hpcm_lq_sr_rd_curves.py \
    --metrics-dir "${METRICS_DIR}" \
    --hpcm-root "${HPCM_ROOT}" \
    --hpcm-hr512-root "/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/hr512/MSE" \
    --hpcm-hr512-metrics-dir "/data/Dataset/LIC-HPCM_outputs/DIV2K_valid100/hr512/rd_metrics" \
    --out-dir "${OUT_ROOT}/rd_curves" \
    --scale-bpp "${SCALE_BPP}"
fi

echo ""
echo "Done."
echo "SR outputs:  ${OUT_ROOT}/baseline/  and  ${OUT_ROOT}/controlnet_canny_e3/"
echo "Metrics:     ${METRICS_DIR}/"
echo "RD curves:   ${OUT_ROOT}/rd_curves/"
