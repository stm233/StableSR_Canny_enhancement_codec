#!/usr/bin/env bash
# StableSR codec environment (self-contained under codec/).

set -euo pipefail

STABLESR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STABLESR_ROOT
export CODEC_ROOT="${STABLESR_ROOT}/codec"

if [[ ! -f "${CODEC_ROOT}/train.py" ]]; then
  echo "Missing HPCM backend at ${CODEC_ROOT}" >&2
  exit 1
fi

export PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"
