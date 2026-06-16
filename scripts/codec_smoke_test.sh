#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")/.." && pwd)/codec/scripts/smoke_test_hpcm.sh" "$@"
