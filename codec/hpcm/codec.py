"""HPCM image codec (backend in codec/src/)."""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn

from codec.paths import ensure_codec_on_path
from codec.utils import crop, get_scale_table, pad

SUPPORTED_MODELS = (
    "HPCM_Base",
    "HPCM_Base_Lite",
    "HPCM_Canny1ch",
    "HPCM_Canny1ch_ME",
    "HPCM_DT1ch",
    "HPCM_Large",
    "HPCM_1B",
)


def build_hpcm_model(model_name: str = "HPCM_Base") -> nn.Module:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name}, choose from {SUPPORTED_MODELS}")
    ensure_codec_on_path()
    mod = importlib.import_module(f"src.models.{model_name}")
    return mod.HPCM()


class HPCMCodec(nn.Module):
    """nn.Module wrapper for StableSR joint training / inference."""

    def __init__(
        self,
        model_name: str = "HPCM_Base",
        checkpoint: str | None = None,
        trainable: bool = False,
        scale_levels: int = 60,
    ):
        super().__init__()
        self.model_name = model_name
        self.scale_levels = scale_levels
        self.codec = build_hpcm_model(model_name)
        if checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            self.codec.update(get_scale_table(0.12, 64, scale_levels))
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        for p in self.codec.parameters():
            p.requires_grad = trainable
        if trainable:
            self.codec.train()
        else:
            self.codec.eval()

    def load_checkpoint(self, checkpoint: str, strict: bool = False) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.codec.load_state_dict(ckpt, strict=strict)
        self.codec.update(get_scale_table(0.12, 64, self.scale_levels))

    def forward(self, x: torch.Tensor, training: bool | None = None) -> dict[str, Any]:
        """x: BxCxHxW in [0,1] (C=1 for Canny1ch, C=3 for RGB/Base)."""
        if training is None:
            training = self.training
        return self.codec(x, training=training)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict[str, Any]:
        h, w = x.size(2), x.size(3)
        out = self.codec.compress(pad(x))
        return {"strings": out["strings"], "shape": out["shape"], "orig_size": (h, w)}

    @torch.no_grad()
    def decompress(self, strings, shape, orig_size: tuple[int, int] | None = None) -> torch.Tensor:
        x_hat = self.codec.decompress(strings, shape)["x_hat"]
        if orig_size is not None:
            x_hat = crop(x_hat, orig_size)
        return x_hat

    def encode_decode(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        return self.forward(x, training=training)["x_hat"]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        model_name: str = "HPCM_Base",
        trainable: bool = False,
    ) -> "HPCMCodec":
        return cls(model_name=model_name, checkpoint=checkpoint, trainable=trainable)
