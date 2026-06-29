"""HQ-VSR_SR_codec: train on canny256, val on canny128, cond = canny64_lossy."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .video_codec_dataset import (
    _l_to_tensor01,
    _load_manifest,
    _prev_canny_rel,
)


def _scan_pairs(
    codec_root: Path,
    target_subdir: str,
    cond_subdir: str,
) -> list[dict]:
    target_root = codec_root / "lossless" / target_subdir
    cond_root = codec_root / cond_subdir
    records = []
    for clip_dir in sorted(p for p in target_root.iterdir() if p.is_dir()):
        clip = clip_dir.name
        for png in sorted(clip_dir.glob("*.png")):
            cond_path = cond_root / clip / png.name
            if not cond_path.is_file():
                continue
            records.append({
                "video": clip,
                "frame": png.stem,
                "target": f"lossless/{target_subdir}/{clip}/{png.name}",
                "cond": str(cond_path.relative_to(codec_root)),
            })
    return records


def _augment_target_cond(
    target: torch.Tensor,
    cond: torch.Tensor,
    patch_size: tuple[int, int] | int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crop/flip on target; cond cropped at H/4 aligned coordinates."""
    if patch_size is None:
        return target, cond
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    ph, pw = patch_size
    _, h, w = target.shape
    if h < ph or w < pw:
        raise ValueError(f"Image {h}x{w} smaller than patch {ph}x{pw}")
    top = random.randint(0, h - ph)
    left = random.randint(0, w - pw)
    target = target[:, top : top + ph, left : left + pw]
    scale_h = max(1, h // cond.shape[-2])
    scale_w = max(1, w // cond.shape[-1])
    ct, cl = top // scale_h, left // scale_w
    cph, cpw = max(1, ph // scale_h), max(1, pw // scale_w)
    cond = cond[:, ct : ct + cph, cl : cl + cpw]
    if random.random() < 0.5:
        target = torch.flip(target, dims=[2])
        cond = torch.flip(cond, dims=[2])
    if random.random() < 0.5:
        target = torch.flip(target, dims=[1])
        cond = torch.flip(cond, dims=[1])
    return target, cond


class HQVSRCondIFrameDataset(Dataset):
    def __init__(
        self,
        codec_root: str | Path,
        target_subdir: str = "canny256",
        cond_subdir: str = "dcvc_lq_qp0/canny64_lossy",
        manifest: str | None = None,
        patch_size: tuple[int, int] | int | None = 256,
        train: bool = True,
        max_samples: int = 0,
        sample_seed: int = 42,
    ):
        self.codec_root = Path(codec_root)
        if manifest:
            self.records = _load_manifest(self.codec_root / manifest)
        else:
            self.records = _scan_pairs(self.codec_root, target_subdir, cond_subdir)
        if max_samples > 0 and len(self.records) > max_samples:
            rng = random.Random(sample_seed)
            self.records = rng.sample(self.records, max_samples)
        self.patch_size = patch_size
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        target = _l_to_tensor01(Image.open(self.codec_root / rec["target"]).convert("L"))
        cond = _l_to_tensor01(Image.open(self.codec_root / rec["cond"]).convert("L"))
        if self.train and self.patch_size is not None:
            target, cond = _augment_target_cond(target, cond, self.patch_size)
        return {"input": target, "cond": cond, "target": target}


class HQVSRCondPFrameDataset(Dataset):
    """P-frame pairs from canny256 manifest + per-frame canny64_lossy cond."""

    def __init__(
        self,
        codec_root: str | Path,
        manifest: str = "manifest_pframe_canny256.jsonl",
        cond_subdir: str = "dcvc_lq_qp0/canny64_lossy",
        patch_size: tuple[int, int] | int | None = 256,
        train: bool = True,
    ):
        self.codec_root = Path(codec_root)
        self.cond_subdir = cond_subdir
        self.records = _load_manifest(self.codec_root / manifest)
        self.patch_size = patch_size
        self.train = train

    def _augment_pframe(
        self,
        p_input: torch.Tensor,
        ref_iframe: torch.Tensor,
        target: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        curr = target
        target, cond = _augment_target_cond(curr, cond, self.patch_size)
        _, h, w = target.shape
        _, oh, ow = curr.shape
        top = (oh - h) // 2 if oh > h else 0
        left = (ow - w) // 2 if ow > w else 0
        p_input = p_input[:, top : top + h, left : left + w]
        ref_iframe = ref_iframe[:, top : top + h, left : left + w]
        return p_input, ref_iframe, target, cond

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        prev_canny = _l_to_tensor01(
            Image.open(self.codec_root / _prev_canny_rel(rec)).convert("L")
        )
        curr_canny = _l_to_tensor01(
            Image.open(self.codec_root / rec["curr_canny"]).convert("L")
        )
        cond = _l_to_tensor01(
            Image.open(self.codec_root / rec["cond"]).convert("L")
        )

        p_input = torch.cat([prev_canny, prev_canny, curr_canny], dim=0)
        ref_iframe = prev_canny.repeat(3, 1, 1)
        target = curr_canny

        if self.train and self.patch_size is not None:
            p_input, ref_iframe, target, cond = self._augment_pframe(
                p_input, ref_iframe, target, cond
            )

        return {
            "input": p_input,
            "ref_iframe": ref_iframe,
            "cond": cond,
            "target": target,
        }


def build_hqvsr_cond_splits(
    codec_root: str | Path,
    val_samples: int = 500,
    val_seed: int = 42,
    patch_size: tuple[int, int] | int = 256,
    stage: str = "iframe",
):
    codec_root = Path(codec_root)
    if stage == "iframe":
        train_ds = HQVSRCondIFrameDataset(
            codec_root, target_subdir="canny256", train=True, patch_size=patch_size
        )
        val_ds = HQVSRCondIFrameDataset(
            codec_root,
            target_subdir="canny128",
            train=False,
            patch_size=None,
            max_samples=val_samples,
            sample_seed=val_seed,
        )
        return train_ds, val_ds

    train_ds = HQVSRCondPFrameDataset(codec_root, train=True, patch_size=patch_size)
    val_ds = HQVSRCondPFrameDataset(
        codec_root,
        manifest="manifest_pframe_canny128.jsonl",
        train=False,
        patch_size=None,
    )
    if len(val_ds) > val_samples:
        rng = random.Random(val_seed)
        indices = rng.sample(range(len(val_ds)), val_samples)
        val_ds.records = [val_ds.records[i] for i in sorted(indices)]
    return train_ds, val_ds
