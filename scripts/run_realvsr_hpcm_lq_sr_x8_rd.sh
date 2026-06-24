#!/usr/bin/env bash
# RealVSR: HPCM_Base decompressed LR64 -> StableSR baseline x8 (no Canny).
# Metrics: PSNR(Y), MS-SSIM(Y), LPIPS vs gt512; bpp@512 = bpp@64 / 64 from HPCM results.
#
# Prereq:
#   bash codec/scripts/hpcm/prepare_realvsr_lr64_flat.sh
#   LAMBDA=all bash codec/scripts/hpcm/test_realvsr_hpcm_base.sh
#
# Usage:
#   LAMBDA=0.0035 bash scripts/run_realvsr_hpcm_lq_sr_x8_rd.sh
#   bash scripts/run_realvsr_hpcm_lq_sr_x8_rd.sh              # all 6 lambdas
#   RUN_INFER=0 bash scripts/run_realvsr_hpcm_lq_sr_x8_rd.sh   # eval + plot only

set -euo pipefail

STABLESR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

HPCM_ROOT="${HPCM_ROOT:-/data/Dataset/LIC-HPCM_outputs/RealVSR_GT_test_lr64/MSE}"
GT_DIR="${GT_DIR:-/data/Dataset/RealVSR_GT_test_iframe_all_flat/gt512}"
OUT_ROOT="${OUT_ROOT:-/data/Dataset/StableSR-TestSets/RealVSR_codec_rd/x8}"
LOG_DIR="${OUT_ROOT}/logs"
METRICS_DIR="${OUT_ROOT}/metrics"

ALL_LAMBDAS=(0.0018 0.0035 0.0067 0.013 0.025 0.0483)
if [[ -n "${LAMBDA:-}" ]]; then
  if [[ "${LAMBDA}" == "all" ]]; then
    LAMBDAS=("${ALL_LAMBDAS[@]}")
  else
    LAMBDAS=("${LAMBDA}")
  fi
else
  LAMBDAS=("${ALL_LAMBDAS[@]}")
fi

BASELINE_CFG="configs/stableSRNew/v2-finetune_text_T_512.yaml"
BASELINE_CKPT="checkpoints/stablesr_000117.ckpt"
VQGAN_CKPT="checkpoints/vqgan_cfw_00011.ckpt"

DDPM_STEPS=100
INPUT_SIZE=512
DEC_W=0.5
SEED=42
SCALE_BPP=64

mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${OUT_ROOT}/baseline"

echo "STABLESR_ROOT=${STABLESR_ROOT}"
echo "HPCM_ROOT=${HPCM_ROOT}"
echo "GT_DIR=${GT_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LAMBDAS=${LAMBDAS[*]}"
echo ""

run_baseline() {
  local lam="$1"
  local lq_dir="${HPCM_ROOT}/lambda_${lam}/images"
  local hpcm_results="${HPCM_ROOT}/lambda_${lam}/results.json"
  local out_dir="${OUT_ROOT}/baseline/MSE_${lam}_x8"
  local log="${LOG_DIR}/baseline_MSE_${lam}_x8.log"
  local metrics_json="${METRICS_DIR}/baseline_MSE_${lam}_x8.json"

  [[ -d "${lq_dir}" ]] || { echo "[SKIP] missing LQ: ${lq_dir}"; return 1; }
  [[ -f "${hpcm_results}" ]] || { echo "[SKIP] missing HPCM results: ${hpcm_results}"; return 1; }

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -d "${out_dir}" ]] && \
       [[ "$(find "${out_dir}" -maxdepth 1 -name '*.png' ! -name '*_canny.png' | wc -l)" -gt 0 ]]; then
      echo "[SKIP infer] baseline MSE_${lam}"
    else
      echo "======== RealVSR baseline | MSE lambda=${lam} | x8 ========"
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

      # Attach LQ codec bpp (bpp@512 = bpp@64 / SCALE_BPP).
      "${PYTHON}" - <<PY
import json
from pathlib import Path
hpcm = json.loads(Path("${hpcm_results}").read_text())
metrics = json.loads(Path("${metrics_json}").read_text())
bpp_64 = float(hpcm["summary"]["bpp"])
metrics["bpp_64"] = bpp_64
metrics["bpp_512"] = bpp_64 / ${SCALE_BPP}
metrics["hpcm_lambda"] = "${lam}"
metrics["lq_dir"] = "${lq_dir}"
Path("${metrics_json}").write_text(json.dumps(metrics, indent=2) + "\n")
print(f"bpp@64={bpp_64:.6f}  bpp@512={bpp_64/${SCALE_BPP}:.6f}")
PY
    fi
  fi
}

for lam in "${LAMBDAS[@]}"; do
  run_baseline "${lam}"
done

if [[ "${RUN_PLOT}" == "1" && "${#LAMBDAS[@]}" -gt 1 ]]; then
  cd "${STABLESR_ROOT}"
  "${PYTHON}" scripts/plot_hpcm_lq_sr_rd_curves.py \
    --metrics-dir "${METRICS_DIR}" \
    --hpcm-root "${HPCM_ROOT}" \
    --out-dir "${OUT_ROOT}/rd_curves" \
    --scale-bpp "${SCALE_BPP}"
fi

echo ""
echo "Done."
echo "SR outputs: ${OUT_ROOT}/baseline/"
echo "Metrics:    ${METRICS_DIR}/baseline_MSE_*_x8.json  (includes bpp_64, bpp_512)"
