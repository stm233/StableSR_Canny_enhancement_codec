"""Dataset: Canny L PNG -> DT RGB (R=dist, G=loc_x, B=loc_y)."""

from __future__ import annotations

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.utils.distance_transform import canny_to_dt_rgb


class CannyDTDataset(Dataset):
    """
    source='canny_l': load binary Canny, augment edge, compute DT [3,H,W].
    source='dt_rgb': load precomputed DT RGB PNG (from prepare_dt_canny_dataset.py).
    Training loss uses R channel (distance) only.
    """

    def __init__(
        self,
        data_path,
        transform=None,
        edge_threshold: float = 0.5,
        source: str = "canny_l",
    ):
        self.data_dir = data_path
        self.transform = transform
        self.edge_threshold = edge_threshold
        self.source = source
        self.dataset_list = sorted(
            f
            for f in os.listdir(self.data_dir)
            if os.path.isfile(os.path.join(self.data_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        )

    def __len__(self):
        return len(self.dataset_list)

    @staticmethod
    def _load_edge01(path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        t = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.size[1], img.size[0])
        return (t / 255.0).unsqueeze(0)

    def _load_dt_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return ToTensor()(img)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.dataset_list[idx])
        if self.source == "dt_rgb":
            rgb = self._load_dt_rgb(path)
            if self.transform is not None:
                rgb = self.transform(rgb)
            return rgb

        edge = self._load_edge01(path)
        if self.transform is not None:
            edge = self.transform(edge)
        return canny_to_dt_rgb(edge, threshold=self.edge_threshold)
