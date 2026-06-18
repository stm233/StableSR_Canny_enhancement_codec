"""Video codec datasets for HPCM two-stage training (I-frame / P-frame)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils.distance_transform import canny_to_dt_rgb


def _load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _l_to_tensor01(img_l: Image.Image) -> torch.Tensor:
    """L-mode [0,255] -> float tensor [1,H,W] in [0,1]."""
    arr = torch.tensor(list(img_l.getdata()), dtype=torch.float32).reshape(img_l.size[1], img_l.size[0]) / 255.0
    return arr.unsqueeze(0)


class _AugmentCropFlip:
    def __init__(self, patch_size: tuple[int, int] | int):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

    def __call__(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        _, h, w = tensors[0].shape
        ph, pw = self.patch_size
        if h < ph or w < pw:
            raise ValueError(f"Image {h}x{w} smaller than patch {ph}x{pw}")
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        cropped = [t[:, top : top + ph, left : left + pw] for t in tensors]
        if random.random() < 0.5:
            cropped = [torch.flip(t, dims=[2]) for t in cropped]
        if random.random() < 0.5:
            cropped = [torch.flip(t, dims=[1]) for t in cropped]
        return cropped


class IFrameDataset(Dataset):
    """I-frame: input R=G=B = binary Canny [3,H,W], loss target is 1ch recon."""

    def __init__(
        self,
        data_root: str | Path,
        manifest: str = "manifest_iframe.jsonl",
        patch_size: tuple[int, int] | int | None = None,
        train: bool = True,
    ):
        self.data_root = Path(data_root)
        self.records = _load_manifest(self.data_root / manifest)
        self.patch_size = patch_size
        self.train = train
        self.augment = _AugmentCropFlip(patch_size) if train and patch_size else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec = self.records[idx]
        canny = Image.open(self.data_root / rec["canny"]).convert("L")
        t = _l_to_tensor01(canny)
        x = t.repeat(3, 1, 1)
        if self.augment is not None:
            x, = self.augment([x])
        return x


def _prev_canny_rel(rec: dict) -> str:
    if "prev_canny" in rec:
        return rec["prev_canny"]
    return rec["prev_gray"].replace("/gray/", "/canny/")


class PFrameDataset(Dataset):
    """P-frame: encoder R=G=prev canny, B=curr canny; ref_encoder on prev canny; target=curr canny."""

    def __init__(
        self,
        data_root: str | Path,
        manifest: str = "manifest_pframe.jsonl",
        patch_size: tuple[int, int] | int | None = None,
        train: bool = True,
    ):
        self.data_root = Path(data_root)
        self.records = _load_manifest(self.data_root / manifest)
        self.patch_size = patch_size
        self.train = train
        self.augment = _AugmentCropFlip(patch_size) if train and patch_size else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        prev_canny = _l_to_tensor01(
            Image.open(self.data_root / _prev_canny_rel(rec)).convert("L")
        )
        curr_canny = _l_to_tensor01(
            Image.open(self.data_root / rec["curr_canny"]).convert("L")
        )

        p_input = torch.cat([prev_canny, prev_canny, curr_canny], dim=0)
        ref_iframe = prev_canny.repeat(3, 1, 1)
        target = curr_canny

        if self.augment is not None:
            p_input, ref_iframe, target = self.augment([p_input, ref_iframe, target])

        return {
            "input": p_input,
            "ref_iframe": ref_iframe,
            "target": target,
        }


class PFrameDTDataset(Dataset):
    """P-frame DT1ch: prev/curr Canny L -> DT RGB encoder in; target = curr Canny."""

    def __init__(
        self,
        data_root: str | Path,
        manifest: str = "manifest_pframe.jsonl",
        patch_size: tuple[int, int] | int | None = None,
        train: bool = True,
    ):
        self.data_root = Path(data_root)
        self.records = _load_manifest(self.data_root / manifest)
        self.patch_size = patch_size
        self.train = train
        self.augment = _AugmentCropFlip(patch_size) if train and patch_size else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        prev_canny = _l_to_tensor01(
            Image.open(self.data_root / _prev_canny_rel(rec)).convert("L")
        )
        curr_canny = _l_to_tensor01(
            Image.open(self.data_root / rec["curr_canny"]).convert("L")
        )

        prev_dt = canny_to_dt_rgb(prev_canny.squeeze(0))
        curr_dt = canny_to_dt_rgb(curr_canny.squeeze(0))

        p_input = torch.cat([prev_dt[0:1], prev_dt[1:2], curr_dt[0:1]], dim=0)
        ref_iframe = prev_dt
        target = curr_canny

        if self.augment is not None:
            p_input, ref_iframe, target = self.augment([p_input, ref_iframe, target])

        return {
            "input": p_input,
            "ref_iframe": ref_iframe,
            "target": target,
        }
