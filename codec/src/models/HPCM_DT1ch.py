"""HPCM DT codec: 3ch DT in (R,G,B), 1ch inverted R out (edge=1, bg=distance)."""

from .HPCM_Canny1ch import HPCM, g_s_1ch  # noqa: F401

__all__ = ["HPCM", "g_s_1ch"]
