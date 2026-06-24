#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Precompute lossy I-frame refs for P-frame DT1ch (unique prev frames only).
#
# Usage:
#   INIT_CKPT=/path/to/HPCM_DT1ch/epoch_best.pth.tar \
#   DATA_ROOT=/data/Dataset/HQ-VSR_processed \
#   CACHE_ROOT=/data/Dataset/LIC-HPCM_outputs/pframe_ref_cache_dt1ch_00155 \
#   bash codec/scripts/hpcm/prepare_pframe_ref_cache.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_processed}"
INIT_CKPT="${INIT_CKPT:-}"
CACHE_ROOT="${CACHE_ROOT:-/data/Dataset/LIC-HPCM_outputs/pframe_ref_cache}"
DEVICE="${DEVICE:-cuda}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.5}"
OVERWRITE="${OVERWRITE:-0}"

[[ -f "${DATA_ROOT}/manifest_pframe.jsonl" ]] || {
  echo "Missing ${DATA_ROOT}/manifest_pframe.jsonl"
  exit 1
}
[[ -n "${INIT_CKPT}" ]] || { echo "Set INIT_CKPT=... (HPCM_DT1ch I-frame ckpt)"; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "Missing INIT_CKPT: ${INIT_CKPT}"; exit 1; }

mkdir -p "${CACHE_ROOT}"
extra=()
[[ "${OVERWRITE}" == "1" ]] && extra+=(--overwrite)

echo "I-frame ckpt: ${INIT_CKPT}"
echo "Data root:    ${DATA_ROOT}"
echo "Cache root:   ${CACHE_ROOT}"
echo "edge_thr:     ${EDGE_THRESHOLD}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" scripts/hpcm/prepare_pframe_ref_cache.py \
  --iframe-checkpoint "${INIT_CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --out-dir "${CACHE_ROOT}" \
  --device "${DEVICE}" \
  --edge-threshold "${EDGE_THRESHOLD}" \
  "${extra[@]}"

echo "Done. Cache: ${CACHE_ROOT}"
