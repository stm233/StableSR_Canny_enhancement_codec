"""HPCM DT codec: 3ch in (R=L1 dist, G=loc_x, B=loc_y), 1ch distance out."""

from .HPCM_Canny1ch import HPCM, g_s_1ch  # noqa: F401

__all__ = ["HPCM", "g_s_1ch"]
