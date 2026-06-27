"""HQ-VSR_SR_codec: per-clip hq512 / lq64_lossy / canny256 triplets for ControlNet SR x8."""

from __future__ import annotations

import json
import random
from pathlib import Path

from torch.utils import data as data

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def _scan_triplets(codec_root: Path, gt_sub: str, lq_sub: str, canny_sub: str) -> list[dict]:
    gt_root = codec_root / gt_sub
    lq_root = codec_root / lq_sub
    canny_root = codec_root / canny_sub
    paths = []
    for clip_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        clip = clip_dir.name
        for gt_path in sorted(clip_dir.glob("*.png")):
            lq_path = lq_root / clip / gt_path.name
            canny_path = canny_root / clip / gt_path.name
            if lq_path.is_file() and canny_path.is_file():
                paths.append(
                    {
                        "gt_path": str(gt_path),
                        "lq_path": str(lq_path),
                        "canny_path": str(canny_path),
                    }
                )
    return paths


def _load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@DATASET_REGISTRY.register(suffix="basicsr")
class HQVSRClipPairedLQCannyDataset(data.Dataset):
    """
    GT:   lossless/hq512/{clip}/{frame}.png          512x512 RGB
    LQ:   dcvc_lq_qp0/lq64_lossy/{clip}/{frame}.png  64x64 RGB  -> upscaled to 512 in training
    Canny: lossless/canny256/{clip}/{frame}.png      256x256 L  -> upscaled to 512 in training
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        codec_root = Path(opt["codec_root"])

        manifest = opt.get("manifest")
        if manifest:
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute():
                manifest_path = codec_root / manifest
            self.paths = _load_manifest(manifest_path)
        else:
            self.paths = _scan_triplets(
                codec_root,
                opt.get("gt_subdir", "lossless/hq512"),
                opt.get("lq_subdir", "dcvc_lq_qp0/lq64_lossy"),
                opt.get("canny_subdir", "lossless/canny256"),
            )

        if opt.get("shuffle_index"):
            rng = random.Random(int(opt.get("seed", 42)))
            rng.shuffle(self.paths)

        if "max_num" in opt:
            self.paths = self.paths[: int(opt["max_num"])]

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No HQ-VSR triplets under codec_root={codec_root} "
                f"(manifest={manifest})"
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)
        rec = self.paths[index]
        img_gt = imfrombytes(self.file_client.get(rec["gt_path"], "gt"), float32=True)
        img_lq = imfrombytes(self.file_client.get(rec["lq_path"], "lq"), float32=True)
        img_canny = imfrombytes(self.file_client.get(rec["canny_path"], "canny"), float32=True)

        if self.opt.get("use_hflip", False):
            img_gt, img_lq, img_canny = augment(
                [img_gt, img_lq, img_canny], True, self.opt.get("use_rot", False)
            )

        img_gt, img_lq, img_canny = img2tensor([img_gt, img_lq, img_canny], bgr2rgb=True, float32=True)

        # ControlNet hint expects 3 channels (L canny -> RGB)
        if img_canny.shape[0] == 1:
            img_canny = img_canny.repeat(3, 1, 1)

        return {
            "gt": img_gt,
            "lq": img_lq,
            "canny": img_canny,
            "gt_path": rec["gt_path"],
            "lq_path": rec["lq_path"],
            "canny_path": rec["canny_path"],
        }

    def __len__(self):
        return len(self.paths)
