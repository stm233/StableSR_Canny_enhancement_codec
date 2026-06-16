#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Download HPCM pretrained checkpoints from Google Drive.
# Usage:
#   bash scripts/download_checkpoints.sh base          # HPCM_Base, all lambdas
#   bash scripts/download_checkpoints.sh large         # HPCM_Large, all lambdas
#   bash scripts/download_checkpoints.sh phi           # HPCM_Phi_Context (lambda=0.013)
#   bash scripts/download_checkpoints.sh base 0.013    # single file

set -euo pipefail

# LIC_ROOT from env.sh
CKPT_ROOT="${LIC_ROOT}/checkpoints"
MODEL="${1:-base}"
LAMBDA="${2:-all}"

mkdir -p "${CKPT_ROOT}"

download() {
  local out_dir="$1" file_id="$2" name="$3"
  mkdir -p "${out_dir}"
  local out="${out_dir}/${name}"
  if [[ -f "${out}" ]]; then
    echo "skip (exists): ${out}"
    return
  fi
  echo "downloading: ${out}"
  gdown "https://drive.google.com/uc?id=${file_id}" -O "${out}"
}

# HPCM_Base MSE
declare -A BASE_MSE=(
  [0.0018]=1nIoANbXzBNE0S_VoLo9ZDHU50lPMmeBP
  [0.0035]=15J_nl33_5R_qyTIzLAaT60ICn9BMGHlB
  [0.0067]=1HIzsEqAPztaMh0Frqec4TtRwoc7uxO97
  [0.013]=1Snq7vkWQdApzCe-gK_V-WuRyMHQRL443
  [0.025]=1NFZD87BkfU28YnDqpzfphG0xDZDZpUA5
  [0.0483]=1G5wm4KENBY2qSAQBxNw3Rz4JcMxH8HXu
)
# HPCM_Base MS-SSIM
declare -A BASE_MSSSIM=(
  [2.4]=1AZ9dY2J9Rn17YSQe_NYIOID-st1C-68O
  [4.58]=1Y8gEL4MRNB-TBbOMDUKeMTO_z1QhbwqL
  [8.73]=1hXK-X6GsjjiULy6FvU80Smob_2UOFeFJ
  [16.64]=1antXt3M0ecOVejbpxL1U7CVx4TS_XPMQ
  [31.73]=1X_Q0hHwAW0GOsHWLoq84YKYqXrduFe6b
  [60.5]=1mX885h4eVwLvpeHpBHBoM1p4Z2VLV2y-
)
# HPCM_Large MSE
declare -A LARGE_MSE=(
  [0.0018]=1E1DUaPsIrfNPwfk4qD-630hhxx5n_BJ4
  [0.0035]=15yDUVvEBn-7dMA9SBIQ2w28LJXBGntQo
  [0.0067]=1yzZKji6RpsyQPD6KFr_weavVrlmn-V4R
  [0.013]=1L19zjwOpbbFPw0FxnyVLcHATxCaorjUV
  [0.025]=1oh8OwCLc8PEVMW1fc9LoC7G4385kHU5D
  [0.0483]=1VWLPQeDzBZgb1D2mZ9jLzLppXL8gUanH
)
# HPCM_Large MS-SSIM
declare -A LARGE_MSSSIM=(
  [2.4]=1RUM2a1wdI8Yj9-tvzO_MnHGZWZRp2-W6
  [4.58]=1TL_QDlfzHvmerN1p0rn5mJbSNwn3LXXx
  [8.73]=1nIEJY9ecr9uA9XidtiQRXQ2rzm1DWKM0
  [16.64]=1sKnWry4LIZPawwv08TH3l_41giUuElCx
  [31.73]=1rR0vFbQ2fOT7EgJbYg5f0OdiIT5jbPPu
  [60.5]=1ITR5JEzLjmdHLp20GYzIdwE8eEK2d7ns
)

download_group() {
  local out_dir="$1"
  shift
  local -n group="$1"
  local key
  for key in "${!group[@]}"; do
    if [[ "${LAMBDA}" == "all" || "${LAMBDA}" == "${key}" ]]; then
      download "${out_dir}" "${group[$key]}" "${key}.pth.tar"
    fi
  done
}

case "${MODEL}" in
  base)
    download_group "${CKPT_ROOT}/HPCM_Base/MSE" BASE_MSE
    download_group "${CKPT_ROOT}/HPCM_Base/MSSSIM" BASE_MSSSIM
    ;;
  large)
    download_group "${CKPT_ROOT}/HPCM_Large/MSE" LARGE_MSE
    download_group "${CKPT_ROOT}/HPCM_Large/MSSSIM" LARGE_MSSSIM
    ;;
  phi)
    download "${CKPT_ROOT}/HPCM_Phi_Context" "1DgH0GQwt4OGQI8EZBxYyucC25yHQv69Q" "0.013.pth.tar"
    ;;
  1b)
    download "${CKPT_ROOT}/HPCM_1B" "1dTtSoIMgyuZ2SYXZz9BrMfN-9m-NH6wM" "0.0483.pth.tar"
    ;;
  *)
    echo "Unknown model: ${MODEL}"
    echo "Usage: $0 {base|large|phi|1b} [lambda]"
    exit 1
    ;;
esac

echo "Checkpoints saved under ${CKPT_ROOT}"
