#!/usr/bin/env bash
# Shared paths for codec/scripts/hpcm/*.sh
CODEC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STABLESR_ROOT="$(cd "${CODEC_ROOT}/.." && pwd)"
LIC_ROOT="${CODEC_ROOT}"
PYTHON="${PYTHON:-/home/exx/miniconda3/envs/stablesr/bin/python}"

# DDP launch: NPROC=1 (default) = single GPU; NPROC=2 = torchrun on 2 local GPUs.
# Multi-node: NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (see train scripts).
run_train() {
  local nproc="${NPROC:-1}"
  if [[ "${nproc}" -gt 1 ]]; then
    "${PYTHON}" -m torch.distributed.run \
      --nnodes="${NNODES:-1}" \
      --nproc_per_node="${nproc}" \
      --node_rank="${NODE_RANK:-0}" \
      --master_addr="${MASTER_ADDR:-127.0.0.1}" \
      --master_port="${MASTER_PORT:-29500}" \
      "$@"
  else
    "${PYTHON}" "$@"
  fi
}
