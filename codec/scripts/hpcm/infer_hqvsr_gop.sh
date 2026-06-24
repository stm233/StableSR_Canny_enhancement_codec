#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# GOP inference: I + N×P on HQ-VSR (DT1ch bitstream, chain decoded refs).
#
# Usage:
#   NUM_P=4 bash codec/scripts/hpcm/infer_hqvsr_gop.sh
#   NUM_P=7 PFRAME_CKPT=.../epoch_best.pth.tar bash codec/scripts/hpcm/infer_hqvsr_gop.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_processed}"
PFRAME_CKPT="${PFRAME_CKPT:-/data/Dataset/LIC-HPCM_outputs/video_pframe_dt_lambda0.00105/checkpoints/HPCM_Video_PFrame_DT1ch_lmbda0.00105/epoch_best.pth.tar}"
IFRAME_CKPT="${IFRAME_CKPT:-}"
NUM_P="${NUM_P:-7}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.5}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_GOPS="${MAX_GOPS_PER_VIDEO:-0}"
RESULTS_DIR="${RESULTS_DIR:-/data/Dataset/LIC-HPCM_outputs/gop_infer_dt}"
OUTDIR="${OUTDIR:-}"

[[ -f "${PFRAME_CKPT}" ]] || { echo "Missing PFRAME_CKPT: ${PFRAME_CKPT}" >&2; exit 1; }

iframe_args=()
if [[ -n "${IFRAME_CKPT}" ]]; then
  iframe_args=(--iframe-checkpoint "${IFRAME_CKPT}")
fi

out_args=()
if [[ -n "${OUTDIR}" ]]; then
  out_args=(--outdir "${OUTDIR}")
fi

echo "GOP: I + ${NUM_P}P"
echo "Data:  ${DATA_ROOT}"
echo "Ckpt:  ${PFRAME_CKPT}"
echo "edge_thr: ${EDGE_THRESHOLD}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" infer_video_gop.py \
  --pframe-checkpoint "${PFRAME_CKPT}" \
  "${iframe_args[@]}" \
  --dataset-root "${DATA_ROOT}" \
  --num-p "${NUM_P}" \
  --edge-threshold "${EDGE_THRESHOLD}" \
  --max-videos "${MAX_VIDEOS}" \
  --max-gops-per-video "${MAX_GOPS}" \
  --results-dir "${RESULTS_DIR}" \
  "${out_args[@]}"
