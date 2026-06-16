#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")/.." && pwd)/codec/scripts/build_hpcm_extensions.sh" "$@"
