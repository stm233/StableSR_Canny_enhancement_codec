#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/RealVSR_GT_test_iframe_all/manifest_iframe.jsonl}"
GT_ROOT="${GT_ROOT:-${DATA_ROOT}/RealVSR_test/GT}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/RealVSR_GT_test_iframe_all_flat}"

[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}"; exit 1; }
[[ -d "${GT_ROOT}" ]] || { echo "Missing GT: ${GT_ROOT}"; exit 1; }

cd "${CODEC_ROOT}"
"${PYTHON}" scripts/hpcm/prepare_realvsr_lr64_flat.py \
  --manifest "${MANIFEST}" \
  --gt-root "${GT_ROOT}" \
  --out-dir "${OUT_DIR}"
