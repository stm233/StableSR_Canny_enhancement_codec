"""StableSR codec package — self-contained under StableSR/codec/."""

from codec.hpcm import HPCMCodec, build_hpcm_model
from codec.paths import (
    codec_root,
    ensure_codec_on_path,
    get_hpcm_backend_root,
    stablesr_root,
)
from codec.utils import crop, get_scale_table, pad

# backward-compatible aliases
get_lic_hpcm_root = get_hpcm_backend_root
ensure_lic_hpcm_on_path = ensure_codec_on_path

__all__ = [
    "HPCMCodec",
    "build_hpcm_model",
    "codec_root",
    "stablesr_root",
    "get_hpcm_backend_root",
    "get_lic_hpcm_root",
    "ensure_codec_on_path",
    "ensure_lic_hpcm_on_path",
    "pad",
    "crop",
    "get_scale_table",
]
