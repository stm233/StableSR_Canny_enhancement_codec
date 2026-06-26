#!/usr/bin/env bash
# Install spconv wheel matching PyTorch CUDA (no source compile).
#
# Usage (stablesr: torch 1.12 + cu113):
#   bash codec/scripts/hpcm/install_spconv.sh
#
# CUDA 12 driver is fine; pick wheel closest to torch.version.cuda.

source "$(dirname "$0")/env.sh"
set -euo pipefail

TORCH_CUDA="$("${PYTHON}" -c "import torch; print(torch.version.cuda or '118')")"
TORCH_CUDA_MAJ="${TORCH_CUDA%%.*}"

case "${TORCH_CUDA_MAJ}" in
  11) WHEEL="spconv-cu118" ;;
  12) WHEEL="spconv-cu120" ;;
  *)  WHEEL="spconv-cu118" ;;
esac

echo "torch cuda: ${TORCH_CUDA}"
echo "Installing ${WHEEL} into $(dirname "${PYTHON}")..."
"${PYTHON}" -m pip install -U "${WHEEL}"
"${PYTHON}" -c "import spconv; print('spconv OK', spconv.__version__)"
