"""Video codec datasets for HPCM two-stage training (I-frame / P-frame)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils.distance_transform import canny_to_dt_rgb


def pframe_ref_cache_key(prev_canny_rel: str) -> str:
    return hashlib.sha256(prev_canny_rel.encode("utf-8")).hexdigest()[:16]


def pframe_ref_cache_path(cache_root: Path, prev_canny_rel: str) -> Path:
    return cache_root / "refs" / f"{pframe_ref_cache_key(prev_canny_rel)}.pt"


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


def _augment_pframe_dt_cached(
    tensors: list[torch.Tensor],
    ref_feats: dict[str, torch.Tensor],
    patch_size: tuple[int, int] | int | None,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """Random crop/flip on full-res tensors and aligned f1/f2/f3."""
    if patch_size is None:
        return tensors, ref_feats

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    ph, pw = patch_size
    _, h, w = tensors[0].shape
    if h < ph or w < pw:
        raise ValueError(f"Image {h}x{w} smaller than patch {ph}x{pw}")

    top = random.randint(0, h - ph)
    left = random.randint(0, w - pw)
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5

    def crop_flip(t: torch.Tensor, scale: int) -> torch.Tensor:
        s_top, s_left = top // scale, left // scale
        s_ph, s_pw = ph // scale, pw // scale
        out = t[:, s_top : s_top + s_ph, s_left : s_left + s_pw]
        if hflip:
            out = torch.flip(out, dims=[2])
        if vflip:
            out = torch.flip(out, dims=[1])
        return out

    out_tensors = [crop_flip(t, 1) for t in tensors]
    out_feats = {
        "f3": crop_flip(ref_feats["f3"], 16),
        "f2": crop_flip(ref_feats["f2"], 8),
        "f1": crop_flip(ref_feats["f1"], 4),
    }
    return out_tensors, out_feats


class IFrameDataset(Dataset):
    """I-frame: 1ch binary Canny [1,H,W] in [0,1]."""

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
        if self.augment is not None:
            t, = self.augment([t])
        return t


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
    """P-frame DT1ch: prev/curr Canny L -> DT RGB encoder in; target = curr inverted R."""

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
        target = curr_dt[0:1]

        if self.augment is not None:
            p_input, ref_iframe, target = self.augment([p_input, ref_iframe, target])

        return {
            "input": p_input,
            "ref_iframe": ref_iframe,
            "target": target,
        }


class PFrameDTCachedDataset(Dataset):
    """P-frame DT1ch with offline I-frame ref cache (prev_r, prev_g, f1/f2/f3)."""

    def __init__(
        self,
        data_root: str | Path,
        cache_root: str | Path,
        manifest: str = "manifest_pframe.jsonl",
        patch_size: tuple[int, int] | int | None = None,
        train: bool = True,
    ):
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        self.records = _load_manifest(self.data_root / manifest)
        self.patch_size = patch_size
        self.train = train

        meta_path = self.cache_root / "cache_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing cache meta: {meta_path}")
        self.cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        prev_rel = _prev_canny_rel(rec)
        cache_path = pframe_ref_cache_path(self.cache_root, prev_rel)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing ref cache for {prev_rel}: {cache_path}")

        cache = torch.load(cache_path, map_location="cpu")
        prev_r_hat = cache["prev_r_hat"]
        prev_g_hat = cache["prev_g_hat"]
        ref_feats = {k: v.clone() for k, v in cache["ref_feats"].items()}

        curr_canny = _l_to_tensor01(
            Image.open(self.data_root / rec["curr_canny"]).convert("L")
        )
        curr_dt = canny_to_dt_rgb(curr_canny.squeeze(0))
        curr_r = curr_dt[0:1]
        target = curr_r

        p_input = torch.cat([prev_r_hat, prev_g_hat, curr_r], dim=0)
        tensors = [p_input, target]
        if self.train and self.patch_size is not None:
            tensors, ref_feats = _augment_pframe_dt_cached(
                tensors, ref_feats, self.patch_size
            )
            p_input, target = tensors

        return {
            "input": p_input,
            "ref_feats": ref_feats,
            "target": target,
        }
