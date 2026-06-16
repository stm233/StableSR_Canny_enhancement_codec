"""Dataset helpers for 1-channel Canny codec."""

from __future__ import annotations

import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CannyLDataset(Dataset):
    """Load Canny PNG (L or RGB) as single-channel tensor [1, H, W] in [0, 1]."""

    def __init__(self, data_path, transform=None):
        self.data_dir = data_path
        self.dataset_list = sorted(
            f for f in os.listdir(self.data_dir)
            if os.path.isfile(os.path.join(self.data_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    @staticmethod
    def _load_l01(path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        t = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.size[1], img.size[0])
        return (t / 255.0).unsqueeze(0)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.dataset_list[idx])
        x = self._load_l01(path)
        if self.transform is not None:
            x = self.transform(x)
        return x


class CannyRGBDataset(CannyLDataset):
    """Canny L-mode -> R=G=B tensor [3, H, W] for g_a (3ch input)."""

    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        return x.repeat(3, 1, 1)

