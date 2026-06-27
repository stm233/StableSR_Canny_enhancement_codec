#!/usr/bin/env bash
# Run StableSR baseline + ControlNet-Canny on DCVC-RT decoded RealVSR LQ.
#
# Prereq:
#   bash /home/exx/Documents/Tianma/DCVC/scripts/export_realvsr_for_stablesr.sh
#   python codec/scripts/hpcm/prepare_realvsr_canny_flat.py --out-dir /data/Dataset/RealVSR_GT_test_iframe_all_flat/canny512
#
# StableSR expects SAME filenames across folders, e.g. 016_00000.png:
#   --init-img  : DCVC decoded LQ 64x64 (rate_XXX/images; upscaled to 512 inside SR)
#   --canny-dir : flat canny RGB    (canny512, for CN model only)
#   --gt-img    : GT 512            (gt512, metrics only)
#
# Usage:
#   bash scripts/run_stablesr_on_dcvc_realvsr.sh
#   RATE_IDX=2 RUN_BASELINE=1 RUN_CANNY=1 bash scripts/run_stablesr_on_dcvc_realvsr.sh

set -eo pipefail

STABLESR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"

DCVC_OUT="${DCVC_OUT:-/data/Dataset/DCVC_RT_outputs/RealVSR_lr64}"
GT_DIR="${GT_DIR:-/data/Dataset/RealVSR_GT_test_iframe_all_flat/gt512}"
CANNY_DIR="${CANNY_DIR:-/data/Dataset/RealVSR_GT_test_iframe_all_flat/canny512}"
SR_OUT_ROOT="${SR_OUT_ROOT:-/data/Dataset/StableSR-TestSets/DCVC_RT_RealVSR}"

RATE_IDX="${RATE_IDX:-all}"   # all | 0 | 1 | ... | 5
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CANNY="${RUN_CANNY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

BASELINE_CFG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
BASELINE_CKPT="checkpoints/stablesr_000117.ckpt"
CN_CFG="configs/stableSRNew/v2-finetune_text_T_512_controlnet_canny.yaml"
CN_CKPT="${CN_CKPT:-/home/exx/Documents/Tianma/StableSR/logs/2026-06-06T00-08-37_stablesr_cn_canny/checkpoints/epoch=000012.ckpt}"
VQGAN_CKPT="checkpoints/vqgan_cfw_00011.ckpt"

INPUT_SIZE=512
DDPM_STEPS=100
DEC_W=0.5

if [[ ! -d "${CANNY_DIR}" ]]; then
  echo "Preparing flat canny -> ${CANNY_DIR}"
  "${PYTHON}" codec/scripts/hpcm/prepare_realvsr_canny_flat.py --out-dir "${CANNY_DIR}"
fi

mapfile -t RATES < <(ls -d "${DCVC_OUT}"/rate_*/images 2>/dev/null | sort)
[[ ${#RATES[@]} -gt 0 ]] || {
  echo "Missing DCVC decoded images. Run:"
  echo "  bash /home/exx/Documents/Tianma/DCVC/scripts/export_realvsr_for_stablesr.sh"
  exit 1
}

run_one_rate() {
  local lq_dir="$1"
  local rate_name
  rate_name="$(basename "$(dirname "${lq_dir}")")"   # rate_000

  if [[ "${RATE_IDX}" != "all" && "${rate_name}" != "rate_$(printf '%03d' "${RATE_IDX}")" ]]; then
    return 0
  fi

  local base_out="${SR_OUT_ROOT}/baseline/${rate_name}"
  local cn_out="${SR_OUT_ROOT}/controlnet_canny/${rate_name}"
  mkdir -p "${SR_OUT_ROOT}/logs"

  echo "===== ${rate_name} | LQ=${lq_dir} ====="

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -n "$(find "${base_out}" -maxdepth 1 -name '*.png' 2>/dev/null | head -1)" ]]; then
      echo "[skip] baseline ${rate_name}"
    else
      mkdir -p "${base_out}"
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/sr_val_ddpm_text_T_vqganfin_old.py \
        --config "${BASELINE_CFG}" \
        --ckpt "${BASELINE_CKPT}" \
        --vqgan_ckpt "${VQGAN_CKPT}" \
        --init-img "${lq_dir}" \
        --outdir "${base_out}" \
        --input_size "${INPUT_SIZE}" \
        --ddpm_steps "${DDPM_STEPS}" \
        --dec_w "${DEC_W}" \
        --colorfix_type adain \
        --n_samples 1 \
        2>&1 | tee "${SR_OUT_ROOT}/logs/baseline_${rate_name}.log"
      "${PYTHON}" scripts/eval_sr_metrics.py \
        --gt-dir "${GT_DIR}" \
        --out-dir "${base_out}" \
        --device cuda \
        --skip-fid \
        --json-out "${base_out}/metrics.json"
    fi
  fi

  if [[ "${RUN_CANNY}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -n "$(find "${cn_out}" -maxdepth 1 -name '*.png' ! -name '*_canny.png' 2>/dev/null | head -1)" ]]; then
      echo "[skip] canny ${rate_name}"
    else
      mkdir -p "${cn_out}"
      cd "${STABLESR_ROOT}"
      "${PYTHON}" scripts/sr_val_ddpm_text_T_vqganfin_hqCanny.py \
        --config "${CN_CFG}" \
        --ckpt "${CN_CKPT}" \
        --vqgan-config configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml \
        --vqgan_ckpt "${VQGAN_CKPT}" \
        --init-img "${lq_dir}" \
        --canny-dir "${CANNY_DIR}" \
        --gt-img "${GT_DIR}" \
        --outdir "${cn_out}" \
        --input_size "${INPUT_SIZE}" \
        --ddpm_steps "${DDPM_STEPS}" \
        --dec_w "${DEC_W}" \
        --colorfix_type adain \
        --n_samples 1 \
        --compute_metrics \
        --save_canny_vis \
        2>&1 | tee "${SR_OUT_ROOT}/logs/canny_${rate_name}.log"
    fi
  fi
}

for lq in "${RATES[@]}"; do
  run_one_rate "${lq}"
done

echo "Done. SR outputs under ${SR_OUT_ROOT}"
