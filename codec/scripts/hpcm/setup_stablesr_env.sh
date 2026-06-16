#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Reuse StableSR conda env for LIC-HPCM (no separate env needed).
# Usage: bash scripts/setup_stablesr_env.sh

set -euo pipefail

STABLESR_ENV="${STABLESR_ENV:-/home/exx/miniconda3/envs/stablesr}"
# LIC_ROOT from env.sh
PYTHON="${STABLESR_ENV}/bin/python"
PIP="${STABLESR_ENV}/bin/pip"

echo "[1/3] Install LIC-HPCM extra deps into stablesr env..."
"${PIP}" install pytorch-msssim pybind11

echo "[2/3] Compile entropy coder (Python 3.9, cp39 .so)..."
cd "${LIC_ROOT}/src/entropy_models/entropy_coders/unbounded_rans"
"${PYTHON}" setup.py build
cp build/lib.linux-x86_64-cpython-39/*.so "${LIC_ROOT}/src/entropy_models/"

echo "[3/3] Verify imports..."
cd "${CODEC_ROOT}"
"${PYTHON}" - <<'PY'
from src.entropy_models.entropy_models import GGM
from src.models.HPCM_Base import HPCM
import pytorch_msssim
print("OK: entropy_models, HPCM_Base, pytorch_msssim")
PY

echo ""
echo "Done. Activate with: conda activate stablesr"
echo "Run test (needs free GPU):"
echo "  cd ${LIC_ROOT}"
echo "  python test.py --model_name HPCM_Base --dataset /data/Dataset/kodim --checkpoint checkpoints/HPCM_Base/0.013.pth.tar"
