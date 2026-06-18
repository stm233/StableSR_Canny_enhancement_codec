#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
set -euo pipefail

# Prepare RealVSR official GT_test.zip into an I-frame codec dataset (ALL frames).
#
# Input:
#   /data/Dataset/_vsr_test_downloads/GT_test.zip  (or set GT_TEST_ZIP)
#
# Output:
#   /data/Dataset/RealVSR_GT_test_iframe_all/manifest_iframe.jsonl
#   /data/Dataset/RealVSR_GT_test_iframe_all/canny/...
#
# Usage:
#   bash codec/scripts/hpcm/prepare_realvsr_gt_test_iframe_all.sh
#   SIZE=512 bash codec/scripts/hpcm/prepare_realvsr_gt_test_iframe_all.sh

DATA_ROOT="${DATA_ROOT:-/data/Dataset}"
ZIP="${GT_TEST_ZIP:-${DATA_ROOT}/_vsr_test_downloads/GT_test.zip}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/RealVSR_GT_test_iframe_all}"
SIZE="${SIZE:-512}"
FRAME_STEP="${FRAME_STEP:-1}"

[[ -f "${ZIP}" ]] || { echo "Missing GT_test.zip: ${ZIP}"; exit 1; }

TMP="$(mktemp -d "${DATA_ROOT}/_vsr_test_downloads/RealVSR_GT_test.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT

echo "Extracting: ${ZIP} -> ${TMP}"
unzip -q "${ZIP}" -d "${TMP}"

# Expect GT_test/<clip>/xxxx.png
GT_ROOT="${TMP}/GT_test"
[[ -d "${GT_ROOT}" ]] || { echo "Missing extracted GT_test/ under ${TMP}"; exit 1; }

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

