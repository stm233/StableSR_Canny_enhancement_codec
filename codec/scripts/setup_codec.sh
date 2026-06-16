#!/usr/bin/env bash
# One-time codec setup: compile C++ entropy extensions.

set -euo pipefail
STABLESR_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "Codec backend: ${STABLESR_ROOT}/codec"
bash "${STABLESR_ROOT}/codec/scripts/build_hpcm_extensions.sh"
echo "Done. Run: bash codec/scripts/smoke_test_hpcm.sh"
