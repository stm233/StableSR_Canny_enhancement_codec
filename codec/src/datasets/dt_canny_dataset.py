"""Dataset: Canny L PNG -> DT RGB encoder input + binary Canny target."""

from __future__ import annotations

import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.distance_transform import canny_to_dt_rgb


class CannyDTDataset(Dataset):
    """
    source='canny_l': load binary Canny from data_dir.
    source='dt_rgb': list files from DT cache dir, load Canny from canny_dir.

    Returns {"input": [3,H,W] DT RGB, "target": [1,H,W] inverted R channel}.
    Loss / decoder target is inverted R (edge=1, background=distance).
    """

    def __init__(
        self,
        data_path,
        transform=None,
        edge_threshold: float = 0.5,
        source: str = "canny_l",
        canny_dir: str | None = None,
    ):
        self.data_dir = data_path
        self.canny_dir = canny_dir
        self.transform = transform
        self.edge_threshold = edge_threshold
        self.source = source
        self.dataset_list = sorted(
            f
            for f in os.listdir(self.data_dir)
            if os.path.isfile(os.path.join(self.data_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        )
        if self.source == "dt_rgb" and not self.canny_dir:
            raise ValueError("canny_dir is required when source='dt_rgb'")

    def __len__(self):
        return len(self.dataset_list)

    @staticmethod
    def _load_edge01(path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        t = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.size[1], img.size[0])
        return (t / 255.0).unsqueeze(0)

    def __getitem__(self, idx):
        fname = self.dataset_list[idx]
        if self.source == "dt_rgb":
            edge_path = os.path.join(self.canny_dir, fname)
        else:
            edge_path = os.path.join(self.data_dir, fname)

        edge = self._load_edge01(edge_path)
        if self.transform is not None:
            edge = self.transform(edge)
        dt_rgb = canny_to_dt_rgb(edge, threshold=self.edge_threshold)
        return {"input": dt_rgb, "target": dt_rgb[0:1]}
