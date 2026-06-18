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
from torch.utils.tensorboard import SummaryWriter

from src.datasets import IFrameDataset, PFrameDataset, PFrameDTDataset
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
        if x_hat.size(1) == 1 and target.size(1) == 3:
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


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def _unpack_batch(batch, stage: str, device: torch.device):
    if stage == "iframe":
        x = batch.to(device)
        return x, x
    model_batch = {
        "input": batch["input"].to(device),
        "ref_iframe": batch["ref_iframe"].to(device),
    }
    target = batch["target"].to(device)
    return model_batch, target


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, global_step, clip_max_norm, stage
):
    model.train()
    device = next(model.parameters()).device
    meters = {k: AverageMeter() for k in ("loss", "bpp_loss", "mse_loss", "psnr", "y_bpp", "z_bpp")}
    t_start = time.time()

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
                print("non-finite norm, skip this batch")
                continue
        optimizer.step()

        for key in meters:
            meters[key].update(out_criterion[key])

        if i % 1000 == 0:
            t_end = time.time() - t_start
            t_start = time.time()
            bs = target.size(0)
            print(
                f"Train epoch {epoch} [{stage}]: ["
                f"{i * bs}/{len(train_dataloader.dataset)} "
                f"({100.0 * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {meters['loss'].avg:.4f} |"
                f"\tMSE: {meters['mse_loss'].avg:.6f} |"
                f"\tPSNR: {meters['psnr'].avg:.3f} |"
                f"\tBpp: {meters['bpp_loss'].avg:.4f} |"
                f"\ty bpp: {meters['y_bpp'].avg:.4f} |"
                f"\tz bpp: {meters['z_bpp'].avg:.4f} |"
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
    if args.stage == "iframe":
        cls = IFrameDataset
    elif args.model_name == "HPCM_Video_PFrame_DT1ch":
        cls = PFrameDTDataset
    else:
        cls = PFrameDataset

    full = cls(args.dataset_root, patch_size=patch, train=True)
    val_size = max(1, int(len(full) * args.val_ratio))
    train_size = len(full) - val_size
    train_set, val_set = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed or 42),
    )

    train_ds = cls(args.dataset_root, patch_size=patch, train=True)
    val_ds = cls(args.dataset_root, patch_size=None, train=False)

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
    parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.model_name is None:
        if args.stage == "pframe":
            args.model_name = "HPCM_Video_PFrame_DT1ch"
        else:
            args.model_name = "HPCM_Canny1ch"
    print(args)

    tag = f"{args.model_name}_{args.stage}_lmbda{args.lmbda}"
    args.log_dir = os.path.join(args.log_dir, tag)
    args.save_path = os.path.join(args.save_path, tag)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset, test_dataset = build_datasets(args)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(test_dataset)}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=min(8, args.num_workers),
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = importlib.import_module(f".{args.model_name}", "src.models").HPCM()
    print(net)
    net = net.to(device)

    if device == "cuda" and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if args.resume:
        target = net.module if isinstance(net, CustomDataParallel) else net
        print(f"Resuming P-frame checkpoint: {args.resume}")
        if hasattr(target, "load_resume_checkpoint"):
            target.load_resume_checkpoint(args.resume, map_location=device)
        else:
            ckpt = torch.load(args.resume, map_location=device)
            target.load_state_dict(ckpt, strict=True)
    elif args.checkpoint or args.p_codec_init:
        target = net.module if isinstance(net, CustomDataParallel) else net
        if args.model_name in ("HPCM_Video_PFrame", "HPCM_Video_PFrame_DT1ch"):
            if args.checkpoint:
                print(f"Loading I-frame checkpoint (ref path only): {args.checkpoint}")
                target.load_iframe_checkpoint(args.checkpoint, map_location=device)
            if args.p_codec_init:
                print(f"Loading P-frame codec init: {args.p_codec_init}")
                target.load_p_codec_checkpoint(args.p_codec_init, map_location=device)
        elif args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            target.load_state_dict(checkpoint, strict=True)

    # LR schedule: same rates as train.py, milestones ÷10 (2750→275, 2850→285, 2950→295)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    writer = SummaryWriter(args.log_dir)

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
    for epoch in range(args.epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_scheduler(epoch)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        global_step = train_one_epoch(
            net, criterion, train_loader, optimizer, epoch, global_step, args.clip_max_norm, args.stage
        )
        loss = test_epoch(epoch, test_loader, net, criterion, writer, args.stage)

        if loss < best_loss:
            best_loss = loss
            print(f"epoch {epoch} is best now!")
            torch.save(net.state_dict(), os.path.join(args.save_path, "epoch_best.pth.tar"))

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            torch.save(net.state_dict(), os.path.join(args.save_path, f"epoch_{epoch}.pth.tar"))


if __name__ == "__main__":
    main(sys.argv[1:])
