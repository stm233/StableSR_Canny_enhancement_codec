#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
set -euo pipefail

# Prepare MVSR4x_test/GT into an I-frame codec dataset (ALL frames).
#
# Input:
#   /data/Dataset/MVSR4x_test/GT  (or set MVSR4X_GT_ROOT)
#
# Output:
#   /data/Dataset/MVSR4x_GT_iframe_all/manifest_iframe.jsonl
#   /data/Dataset/MVSR4x_GT_iframe_all/canny/...
#
# Usage:
#   bash codec/scripts/hpcm/prepare_mvsr4x_gt_iframe_all.sh

DATA_ROOT="${DATA_ROOT:-/data/Dataset}"
GT_ROOT="${MVSR4X_GT_ROOT:-${DATA_ROOT}/MVSR4x_test/GT}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/MVSR4x_GT_iframe_all}"
SIZE="${SIZE:-512}"
FRAME_STEP="${FRAME_STEP:-1}"

[[ -d "${GT_ROOT}" ]] || { echo "Missing GT root: ${GT_ROOT}"; exit 1; }

echo "Preparing I-frame canny (ALL frames)."
echo "  GT_ROOT=${GT_ROOT}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  SIZE=${SIZE}  FRAME_STEP=${FRAME_STEP}"
cd "${CODEC_ROOT}"
"${PYTHON}" scripts/hpcm/prepare_vsr_iframe_from_frames.py \
  --gt-root "${GT_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --size "${SIZE}" \
  --frame-step "${FRAME_STEP}"

echo "Done: ${OUT_DIR}"

