"""Dataset helpers for 1-channel Canny codec."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _collect_image_paths(data_dir: str | Path) -> list[str]:
    """Flat folder first; if empty, recurse into video subdirs (e.g. RealVSR/canny/016/)."""
    root = Path(data_dir)
    if not root.is_dir():
        return []

    flat = sorted(
        str(p)
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if flat:
        return flat

    return sorted(
        str(p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


class CannyLDataset(Dataset):
    """Load Canny PNG (L or RGB) as single-channel tensor [1, H, W] in [0, 1]."""

    def __init__(self, data_path, transform=None):
        self.data_dir = data_path
        self.dataset_list = _collect_image_paths(data_path)
        if len(self.dataset_list) == 0:
            raise RuntimeError(f"No Canny images found under {data_path}")
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    @staticmethod
    def _load_l01(path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        t = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.size[1], img.size[0])
        return (t / 255.0).unsqueeze(0)

    def __getitem__(self, idx):
        path = self.dataset_list[idx]
        x = self._load_l01(path)
        if self.transform is not None:
            x = self.transform(x)
        return x


class CannyRGBDataset(CannyLDataset):
    """Canny L-mode -> R=G=B tensor [3, H, W] for g_a (3ch input)."""

    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        return x.repeat(3, 1, 1)

