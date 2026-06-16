"""Paths for self-contained codec package under StableSR/codec/."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_MARKER = "train.py"


def stablesr_root() -> Path:
    return Path(__file__).resolve().parents[1]


def codec_root() -> Path:
    """StableSR/codec — contains src/, train.py, hpcm/, scripts/."""
    return Path(__file__).resolve().parent


def get_hpcm_backend_root() -> Path:
    """HPCM backend root (same as codec_root)."""
    root = codec_root()
    if not (root / _BACKEND_MARKER).is_file():
        raise FileNotFoundError(f"Missing HPCM backend at {root} ({_BACKEND_MARKER})")
    return root


# backward-compatible alias
get_lic_hpcm_root = get_hpcm_backend_root


def ensure_codec_on_path() -> Path:
    root = get_hpcm_backend_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ensure_lic_hpcm_on_path = ensure_codec_on_path
