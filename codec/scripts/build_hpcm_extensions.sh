#!/usr/bin/env bash
# Build HPCM C++ entropy extensions (codec/src/entropy_models).

set -euo pipefail
source "$(dirname "$0")/codec_env.sh"

RANS_DIR="${CODEC_ROOT}/src/entropy_models/entropy_coders/unbounded_rans"
DEST="${CODEC_ROOT}/src/entropy_models"

echo "Building entropy coder in ${RANS_DIR}"
"${PYTHON}" -m pip install -q pybind11 pytorch-msssim 2>/dev/null || true

cd "${RANS_DIR}"
"${PYTHON}" setup.py build clean 2>/dev/null || "${PYTHON}" setup.py build
PY_VER="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
SO_DIR="${RANS_DIR}/build/lib.linux-x86_64-cpython-${PY_VER}"
if [[ ! -d "${SO_DIR}" ]]; then
  SO_DIR="$(find build -maxdepth 1 -type d -name "lib.linux-*cpython-${PY_VER}*" | head -1)"
fi
if [[ -z "${SO_DIR}" || ! -d "${SO_DIR}" ]]; then
  echo "Cannot find build output for Python ${PY_VER}" >&2
  exit 1
fi
cp "${SO_DIR}"/*.so "${DEST}/"
echo "Installed .so -> ${DEST}"

cd "${STABLESR_ROOT}"
"${PYTHON}" - <<PY
import sys
sys.path.insert(0, "${CODEC_ROOT}")
from src.models.HPCM_Base import HPCM
print("OK: codec extensions")
PY
