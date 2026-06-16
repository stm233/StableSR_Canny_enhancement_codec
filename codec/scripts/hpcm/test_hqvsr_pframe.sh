#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Test P-frame on fixed HQ-VSR_test500 (500 consecutive pairs).
#
# Usage:
#   PFRAME_CKPT=/path/to/epoch_best.pth.tar bash scripts/test_hqvsr_pframe.sh

set -euo pipefail

# LIC_ROOT from env.sh


DATA_ROOT="${DATA_ROOT:-/data/Dataset/HQ-VSR_test500}"
PFRAME_CKPT="${PFRAME_CKPT:-/data/Dataset/LIC-HPCM_outputs/video_pframe_lambda0.00105/checkpoints/HPCM_Video_PFrame_pframe_lmbda0.00105/epoch_best.pth.tar}"
DEVICE="${DEVICE:-cuda}"

OUT_ROOT="${OUT_ROOT:-/data/Dataset/LIC-HPCM_outputs/HQ-VSR_test500_pframe}"
RESULTS_DIR="${OUT_ROOT}/metrics"

[[ -f "${DATA_ROOT}/manifest_pframe.jsonl" ]] || {
  echo "Missing ${DATA_ROOT}. Run: python scripts/create_hqvsr_test500.py"
  exit 1
}
[[ -f "${PFRAME_CKPT}" ]] || {
  echo "Missing P-frame ckpt: ${PFRAME_CKPT}"
  exit 1
}

mkdir -p "${RESULTS_DIR}"
echo "Test set:   ${DATA_ROOT} (500 fixed P-frame pairs)"
echo "Checkpoint: ${PFRAME_CKPT}"
echo ""

cd "${CODEC_ROOT}"
"${PYTHON}" test_video_pframe.py \
  --checkpoint "${PFRAME_CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --manifest manifest_pframe.jsonl \
  --device "${DEVICE}" \
  --results_dir "${RESULTS_DIR}" \
  2>&1 | tee "${OUT_ROOT}/test.log"
