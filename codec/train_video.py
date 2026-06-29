"""Two-stage video codec training for HPCM (I-frame / P-frame).

Does not modify train.py or existing model classes.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from ddp_utils import (
    barrier,
    cleanup_distributed,
    is_main_process,
    reduce_scalar,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)
from src.datasets import (
    IFrameDataset,
    PFrameDataset,
    PFrameDTDataset,
    PFrameDTCachedDataset,
    build_hqvsr_cond_splits,
)
from utils import psnr_continuous


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda: float = 1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        n, _, h, w = target.size()
        num_pixels = n * h * w
        out = {}
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (
            -math.log(2) * num_pixels
        )
        out["z_bpp"] = torch.log(output["likelihoods"]["z"]).sum() / (
            -math.log(2) * num_pixels
        )
        x_hat = output["x_hat"]
        if x_hat.size(1) == 1 and target.size(1) == 1:
            target_dist = target
        elif x_hat.size(1) == 1 and target.size(1) == 3:
            target_dist = target[:, :1, :, :]
        else:
            target_dist = target
        out["mse_loss"] = self.mse(x_hat, target_dist)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        out["psnr"] = psnr_continuous(x_hat, target_dist, peak=255.0)
        return out


class AverageMeter:
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _is_cond_model(model_name: str) -> bool:
    return model_name.endswith("_Cond")


def _unpack_batch(batch, stage: str, device: torch.device):
    if stage == "iframe":
        if isinstance(batch, dict):
            model_batch = {
                "input": batch["input"].to(device),
                "cond": batch["cond"].to(device),
            }
            return model_batch, batch["target"].to(device)
        x = batch.to(device)
        return x, x
    model_batch = {"input": batch["input"].to(device)}
    if "cond" in batch:
        model_batch["cond"] = batch["cond"].to(device)
    if "ref_feats" in batch:
        model_batch["ref_feats"] = {
            k: v.to(device) for k, v in batch["ref_feats"].items()
        }
    else:
        model_batch["ref_iframe"] = batch["ref_iframe"].to(device)
    target = batch["target"].to(device)
    return model_batch, target


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    epoch,
    global_step,
    clip_max_norm,
    stage,
    *,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
):
    model.train()
    device = next(model.parameters()).device
    meters = {k: AverageMeter() for k in ("loss", "bpp_loss", "mse_loss", "psnr", "y_bpp", "z_bpp")}
    t_start = time.time()
    log_interval = 1000

    for i, batch in enumerate(train_dataloader):
        global_step += 1
        x, target = _unpack_batch(batch, stage, device)
        optimizer.zero_grad()
        out_net = model(x)
        out_criterion = criterion(out_net, target)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if total_norm.isnan() or total_norm.isinf():
                if is_main_process(rank):
                    print("non-finite norm, skip this batch")
                continue
        optimizer.step()

        for key in meters:
            meters[key].update(out_criterion[key])

        if is_main_process(rank) and i % log_interval == 0:
            t_end = time.time() - t_start
            t_start = time.time()
            bs = target.size(0)
            loss_avg = reduce_scalar(meters["loss"].avg, device, is_distributed)
            mse_avg = reduce_scalar(meters["mse_loss"].avg, device, is_distributed)
            psnr_avg = reduce_scalar(meters["psnr"].avg, device, is_distributed)
            bpp_avg = reduce_scalar(meters["bpp_loss"].avg, device, is_distributed)
            y_bpp_avg = reduce_scalar(meters["y_bpp"].avg, device, is_distributed)
            z_bpp_avg = reduce_scalar(meters["z_bpp"].avg, device, is_distributed)
            seen = min(i * bs * world_size, len(train_dataloader.dataset))
            print(
                f"Train epoch {epoch} [{stage}]: ["
                f"{seen}/{len(train_dataloader.dataset)} "
                f"({100.0 * i / max(len(train_dataloader), 1):.0f}%)]"
                f"\tLoss: {loss_avg:.4f} |"
                f"\tMSE: {mse_avg:.6f} |"
                f"\tPSNR: {psnr_avg:.3f} |"
                f"\tBpp: {bpp_avg:.4f} |"
                f"\ty bpp: {y_bpp_avg:.4f} |"
                f"\tz bpp: {z_bpp_avg:.4f} |"
                f"\ttime: {t_end:.2f}"
            )
            torch.cuda.empty_cache()

    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, writer, stage):
    model.eval()
    device = next(model.parameters()).device
    meters = {k: AverageMeter() for k in ("loss", "bpp_loss", "mse_loss", "psnr", "y_bpp", "z_bpp")}

    with torch.no_grad():
        for batch in test_dataloader:
            x, target = _unpack_batch(batch, stage, device)
            out_net = model(x)
            out_criterion = criterion(out_net, target)
            for key in meters:
                meters[key].update(out_criterion[key])

    print(
        f"Test epoch {epoch} [{stage}]:"
        f"\tLoss: {meters['loss'].avg:.4f} |"
        f"\tMSE: {meters['mse_loss'].avg:.6f} |"
        f"\tPSNR: {meters['psnr'].avg:.3f} |"
        f"\tBpp: {meters['bpp_loss'].avg:.4f} |"
        f"\ty bpp: {meters['y_bpp'].avg:.4f} |"
        f"\tz bpp: {meters['z_bpp'].avg:.4f}"
    )
    writer.add_scalar("test_loss", meters["loss"].avg, global_step=epoch)
    writer.add_scalar("test_mse_loss", meters["mse_loss"].avg, global_step=epoch)
    writer.add_scalar("test_bpp_loss", meters["bpp_loss"].avg, global_step=epoch)
    return meters["loss"].avg


def build_datasets(args):
    patch = tuple(args.patch_size)
    if getattr(args, "hqvsr_codec", False):
        return build_hqvsr_cond_splits(
            args.dataset_root,
            val_samples=args.val_samples,
            val_seed=args.seed or 42,
            patch_size=patch,
            stage=args.stage,
        )

    if args.stage == "iframe":
        cls = IFrameDataset
        cls_kwargs = {}
    elif args.model_name == "HPCM_Video_PFrame_DT1ch":
        if getattr(args, "pframe_cache_dir", ""):
            cls = PFrameDTCachedDataset
            cls_kwargs = {"cache_root": args.pframe_cache_dir}
        else:
            cls = PFrameDTDataset
            cls_kwargs = {}
    else:
        cls = PFrameDataset
        cls_kwargs = {}

    full = cls(args.dataset_root, patch_size=patch, train=True, **cls_kwargs)
    val_size = max(1, int(len(full) * args.val_ratio))
    train_size = len(full) - val_size
    train_set, val_set = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed or 42),
    )

    train_ds = cls(args.dataset_root, patch_size=patch, train=True, **cls_kwargs)
    val_ds = cls(args.dataset_root, patch_size=None, train=False, **cls_kwargs)

    return Subset(train_ds, train_set.indices), Subset(val_ds, val_set.indices)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="HPCM video codec training (I/P frame).")
    parser.add_argument("--stage", choices=["iframe", "pframe"], required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/data/Dataset/HQ-VSR_processed",
    )
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument(
        "--hqvsr-codec",
        action="store_true",
        help="Use HQ-VSR_SR_codec: train canny256, val 500 random canny128, cond=canny64_lossy",
    )
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3001)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.00105)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--save_path", type=str, default="/data/Dataset/LIC-HPCM_outputs/video")
    parser.add_argument("--log_dir", type=str, default="/data/Dataset/LIC-HPCM_outputs/video_logs")
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="", help="I-frame ckpt for pframe; full ckpt for iframe")
    parser.add_argument(
        "--p-codec-init",
        type=str,
        default="",
        help="Optional separate init for P-frame main codec (not I-frame ckpt)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume full P-frame checkpoint (HPCM_Video_PFrame state_dict)",
    )
    parser.add_argument(
        "--pframe-cache-dir",
        type=str,
        default="",
        help="P-frame DT1ch: offline I-frame ref cache from prepare_pframe_ref_cache.py",
    )
    parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True)
    return parser.parse_args(argv)


def _resolve_device(args, local_rank: int, is_distributed: bool) -> torch.device:
    if is_distributed:
        return torch.device(f"cuda:{local_rank}")
    if args.cuda and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(
                "WARNING: Multiple GPUs visible but DDP is off. "
                "Use NPROC>1 with torchrun for multi-GPU training. Using cuda:0."
            )
        return torch.device("cuda:0")
    return torch.device("cpu")


def main(argv):
    args = parse_args(argv)
    rank, local_rank, world_size, is_distributed = setup_distributed()
    if args.model_name is None:
        if args.stage == "pframe":
            args.model_name = (
                "HPCM_Video_PFrame_Canny1ch_Spconv_Cond"
                if args.hqvsr_codec
                else "HPCM_Video_PFrame_Canny1ch"
            )
        else:
            args.model_name = (
                "HPCM_Canny1ch_Spconv_Cond" if args.hqvsr_codec else "HPCM_Canny1ch"
            )
    if is_main_process(rank):
        print(args)
        if is_distributed:
            print(f"DDP: rank={rank}/{world_size}, local_rank={local_rank}")
            print(f"Effective batch size: {args.batch_size * world_size}")

    tag = f"{args.model_name}_{args.stage}_lmbda{args.lmbda}"
    if getattr(args, "pframe_cache_dir", ""):
        tag += "_cached"
    args.log_dir = os.path.join(args.log_dir, tag)
    args.save_path = os.path.join(args.save_path, tag)
    if is_main_process(rank):
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)
    barrier(is_distributed)

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)

    train_dataset, test_dataset = build_datasets(args)
    if is_main_process(rank):
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(test_dataset)}")

    device = _resolve_device(args, local_rank, is_distributed)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=min(8, args.num_workers),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    net = importlib.import_module(f".{args.model_name}", "src.models").HPCM()
    if is_main_process(rank):
        print(net)
    net = net.to(device)

    if args.resume:
        if is_main_process(rank):
            print(f"Resuming P-frame checkpoint: {args.resume}")
        if hasattr(net, "load_resume_checkpoint"):
            net.load_resume_checkpoint(args.resume, map_location=device)
        else:
            ckpt = torch.load(args.resume, map_location=device)
            net.load_state_dict(ckpt, strict=True)
    elif args.checkpoint or args.p_codec_init:
        if args.model_name in (
            "HPCM_Video_PFrame",
            "HPCM_Video_PFrame_DT1ch",
            "HPCM_Video_PFrame_Canny1ch",
            "HPCM_Video_PFrame_Canny1ch_ME",
            "HPCM_Video_PFrame_Canny1ch_Spconv",
            "HPCM_Video_PFrame_Canny1ch_Spconv_Cond",
        ):
            if args.checkpoint:
                if is_main_process(rank):
                    print(f"Loading I-frame checkpoint (ref path only): {args.checkpoint}")
                net.load_iframe_checkpoint(args.checkpoint, map_location=device)
            if args.p_codec_init:
                if is_main_process(rank):
                    print(f"Loading P-frame codec init: {args.p_codec_init}")
                net.load_p_codec_checkpoint(args.p_codec_init, map_location=device)
        elif args.checkpoint:
            if is_main_process(rank):
                print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            net.load_state_dict(checkpoint, strict=True)

    barrier(is_distributed)
    if is_distributed:
        find_unused = args.stage == "pframe"
        net = wrap_ddp(net, local_rank, find_unused_parameters=find_unused)

    # LR schedule: same rates as train.py, milestones ÷10 (2750→275, 2850→285, 2950→295)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    writer = SummaryWriter(args.log_dir) if is_main_process(rank) else None

    def lr_scheduler(epoch):
        if epoch < 275:
            return 1e-4
        if epoch < 285:
            return 3e-5
        if epoch < 295:
            return 1e-5
        return 1e-6

    best_loss = float("inf")
    global_step = 0
    try:
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(epoch)
            if is_main_process(rank):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

            global_step = train_one_epoch(
                net,
                criterion,
                train_loader,
                optimizer,
                epoch,
                global_step,
                args.clip_max_norm,
                args.stage,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed,
            )

            if is_main_process(rank):
                loss = test_epoch(epoch, test_loader, net, criterion, writer, args.stage)
                if loss < best_loss:
                    best_loss = loss
                    print(f"epoch {epoch} is best now!")
                    torch.save(
                        unwrap_model(net).state_dict(),
                        os.path.join(args.save_path, "epoch_best.pth.tar"),
                    )
                if args.save_interval > 0 and epoch % args.save_interval == 0:
                    torch.save(
                        unwrap_model(net).state_dict(),
                        os.path.join(args.save_path, f"epoch_{epoch}.pth.tar"),
                    )
            barrier(is_distributed)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main(sys.argv[1:])
