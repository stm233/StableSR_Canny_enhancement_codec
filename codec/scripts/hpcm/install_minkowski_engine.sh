#!/usr/bin/env bash
# Build/install MinkowskiEngine into the stablesr conda env (required for HPCM_*_ME).
#
# Prereqs:
#   - NVIDIA driver (CUDA 12.x driver is OK — backward compatible)
#   - nvcc toolkit should match PyTorch CUDA (stablesr: torch 1.12 + cuda 11.3), NOT driver CUDA 12.0
#   - conda env activated: conda activate stablesr
#
# If nvidia-smi shows CUDA 12.0 but torch.version.cuda is 11.3:
#   conda install -y -c nvidia cuda-toolkit=11.8
#   export CUDA_HOME=$CONDA_PREFIX
#
# Usage:
#   bash codec/scripts/hpcm/install_minkowski_engine.sh
#   CUDA_HOME=/usr/local/cuda-11.8 bash codec/scripts/hpcm/install_minkowski_engine.sh

source "$(dirname "$0")/env.sh"
set -euo pipefail

ME_VERSION="${ME_VERSION:-0.5.4}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for d in /usr/local/cuda-11.8 /usr/local/cuda-11.3 /usr/local/cuda /opt/cuda; do
    if [[ -x "${d}/bin/nvcc" ]]; then
      CUDA_HOME="${d}"
      break
    fi
  done
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA_HOME not set and nvcc not found."
  echo ""
  echo "Driver CUDA 12.x (nvidia-smi) is fine. You still need an 11.x toolkit for torch 1.12+cu113."
  echo "Recommended (inside stablesr env):"
  echo "  conda install -y -c nvidia cuda-toolkit=11.8"
  echo "  export CUDA_HOME=\$CONDA_PREFIX"
  echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
  echo "  bash codec/scripts/hpcm/install_minkowski_engine.sh"
  echo ""
  echo "Or if system has cuda-11.8:"
  echo "  export CUDA_HOME=/usr/local/cuda-11.8"
  exit 1
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "Python:    ${PYTHON}"
echo "CUDA_HOME: ${CUDA_HOME}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvcc --version | tail -1
TORCH_CUDA="$("${PYTHON}" -c "import torch; print(torch.version.cuda or '')")"
NVCC_VER="$(nvcc --version | grep -oP 'release \K[0-9]+' | head -1)"
if [[ -n "${TORCH_CUDA}" && -n "${NVCC_VER}" ]]; then
  TORCH_MAJ="${TORCH_CUDA%%.*}"
  if [[ "${NVCC_VER}" != "${TORCH_MAJ}" && "${NVCC_VER}" -gt 11 ]]; then
    echo ""
    echo "WARN: nvcc is CUDA ${NVCC_VER}.x but PyTorch was built with CUDA ${TORCH_CUDA}."
    echo "      Use CUDA toolkit 11.8 for compilation (see script header), not system CUDA 12."
  fi
fi

echo ""
echo "Installing MinkowskiEngine==${ME_VERSION} (compile from source, may take several minutes)..."
"${PYTHON}" -m pip install -U pip setuptools wheel
"${PYTHON}" -m pip install "MinkowskiEngine==${ME_VERSION}" --no-build-isolation -v

echo ""
"${PYTHON}" -c "import MinkowskiEngine as ME; print('MinkowskiEngine OK', ME.__version__)"
