#!/usr/bin/env bash
# Shared paths for codec/scripts/hpcm/*.sh
CODEC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STABLESR_ROOT="$(cd "${CODEC_ROOT}/.." && pwd)"
LIC_ROOT="${CODEC_ROOT}"
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"
