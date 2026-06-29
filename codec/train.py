import argparse
import math
import random
import sys
import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
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
from utils import psnr_continuous
from src.datasets.canny_dataset import CannyLDataset, CannyRGBDataset
from src.datasets.dt_canny_dataset import CannyDTDataset

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform):
        self.data_dir = data_path
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx])
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img


def build_datasets(args):
    """HPCM_Canny1ch: 1ch L-mode Canny; HPCM_DT1ch: 3ch DT; legacy RGB for other models."""
    model = getattr(args, "model_name", "")
    if model in ("HPCM_Canny1ch", "HPCM_Canny1ch_ME", "HPCM_Canny1ch_Spconv"):
        train_tf = transforms.Compose([
            transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        test_tf = None
        train_dataset = CannyLDataset(args.train_dataset, transform=train_tf)
        test_dataset = CannyLDataset(args.test_dataset, transform=test_tf)
    elif model == "HPCM_DT1ch":
        dt_source = getattr(args, "dt_source", "canny_l")
        train_tf = transforms.Compose([
            transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        test_tf = None
        canny_train = getattr(args, "canny_dataset", "") or args.train_dataset
        canny_test = getattr(args, "canny_test_dataset", "") or args.test_dataset
        train_dataset = CannyDTDataset(
            args.train_dataset,
            transform=train_tf,
            source=dt_source,
            canny_dir=canny_train if dt_source == "dt_rgb" else None,
        )
        test_dataset = CannyDTDataset(
            args.test_dataset,
            transform=test_tf,
            source=dt_source,
            canny_dir=canny_test if dt_source == "dt_rgb" else None,
        )
    else:
        train_dataset = Dataset(
            args.train_dataset,
            transform=transforms.Compose([
                transforms.RandomCrop(args.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]),
        )
        test_dataset = Dataset(
            args.test_dataset,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    return train_dataset, test_dataset

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out['y_bpp'] = torch.log(output['likelihoods']['y']).sum() / (-math.log(2) * num_pixels)
        out['z_bpp'] = torch.log(output['likelihoods']['z']).sum() / (-math.log(2) * num_pixels)
        x_hat = output["x_hat"]
        if x_hat.size(1) == 1 and target.size(1) == 1:
            target_dist = target
        elif x_hat.size(1) == 1 and target.size(1) == 3:
            # HPCM_Canny1ch: R=G=B repeated canny
            target_dist = target[:, :1, :, :]
        else:
            target_dist = target
        out["mse_loss"] = self.mse(x_hat, target_dist)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        out["psnr"] = psnr_continuous(x_hat, target_dist, peak=255.0)

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # [FIX] Get Python float to prevent graph leak
        if isinstance(val, torch.Tensor):
            val = val.detach().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    epoch,
    global_step,
    clip_max_norm,
    model_name="",
    *,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
):
    model.train()
    if is_main_process(rank):
        print(model.training)
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    t_start = time.time()
    for i, batch in enumerate(train_dataloader):

        global_step+=1
        x, target = _unpack_batch(batch, model_name, device)
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

        bpp_loss.update(out_criterion["bpp_loss"])
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])
        psnr.update(out_criterion["psnr"])
        y_bpp.update(out_criterion["y_bpp"])
        z_bpp.update(out_criterion["z_bpp"])

        if is_main_process(rank) and i % 100 == 0 :
            t_end = time.time()-t_start
            t_start = time.time()
            loss_avg = reduce_scalar(loss.avg, device, is_distributed)
            mse_avg = reduce_scalar(mse_loss.avg, device, is_distributed)
            psnr_avg = reduce_scalar(psnr.avg, device, is_distributed)
            bpp_avg = reduce_scalar(bpp_loss.avg, device, is_distributed)
            y_bpp_avg = reduce_scalar(y_bpp.avg, device, is_distributed)
            z_bpp_avg = reduce_scalar(z_bpp.avg, device, is_distributed)
            seen = min(i * len(x) * world_size, len(train_dataloader.dataset))
            print(
                f"Train epoch {epoch}: ["
                f"{seen}/{len(train_dataloader.dataset)}"
                f" ({100. * i / max(len(train_dataloader), 1):.0f}%)]"
                f"\tLoss: {loss_avg:.4f} |"
                f"\tMSE loss: {mse_avg:.6f} |"
                f"\tPSNR: {psnr_avg:.3f} |"
                f"\tBpp loss: {bpp_avg:.4f} |"
                f"\ty bpp: {y_bpp_avg:.4f} |"
                f"\tz bpp: {z_bpp_avg:.4f} |"
                f'\t time : {t_end:.2f} |'
            )
            torch.cuda.empty_cache()
        
    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, writer, model_name=""):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    with torch.no_grad():
        for batch in test_dataloader:
            x, target = _unpack_batch(batch, model_name, device)
            out_net = model(x)
            out_criterion = criterion(out_net, target)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion["psnr"])
            y_bpp.update(out_criterion["y_bpp"])
            z_bpp.update(out_criterion["z_bpp"])
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tPSNR: {psnr.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty bpp: {y_bpp.avg:.4f} |"
        f"\tz bpp: {z_bpp.avg:.4f} |"
    )
    writer.add_scalar("test_loss", loss.avg, global_step = epoch)
    writer.add_scalar("test_mse_loss", mse_loss.avg, global_step = epoch)
    writer.add_scalar("test_bpp_loss", bpp_loss.avg, global_step = epoch)

    return loss.avg

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_class", type=str, default="hypers")
    parser.add_argument(
        "-tr_d", "--train_dataset", type=str, help="Training dataset"
    )
    parser.add_argument(
        "-te_d", "--test_dataset", type=str, help="Testing dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=3001,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.013,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="/output/", help="Where to Save model"
    )
    parser.add_argument(
        "--log_dir", type=str, default="/output/", help="Where to Save logs"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N epochs (0 = disable periodic saves)",
    )
    parser.add_argument(
        "--dt-source",
        dest="dt_source",
        choices=["canny_l", "dt_rgb"],
        default="canny_l",
        help="HPCM_DT1ch: canny_l=on-the-fly DT, dt_rgb=precomputed RGB cache",
    )
    parser.add_argument(
        "--canny-dataset",
        dest="canny_dataset",
        type=str,
        default="",
        help="HPCM_DT1ch dt_rgb: Canny L dir for loss target (train)",
    )
    parser.add_argument(
        "--canny-test-dataset",
        dest="canny_test_dataset",
        type=str,
        default="",
        help="HPCM_DT1ch dt_rgb: Canny L dir for loss target (test)",
    )
    args = parser.parse_args(argv)
    return args


def _unpack_batch(batch, model_name: str, device):
    if model_name == "HPCM_DT1ch":
        x = batch["input"].to(device)
        target = batch["target"].to(device)
        return x, target
    x = batch.to(device)
    return x, x


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
    if is_main_process(rank):
        print(args)
        if is_distributed:
            print(f"DDP: rank={rank}/{world_size}, local_rank={local_rank}")
            print(f"Effective batch size: {args.batch_size * world_size}")
    args.log_dir = os.path.join(args.log_dir, args.model_name + '_lmbda' + str(args.lmbda))
    args.save_path = os.path.join(args.save_path, args.model_name + '_lmbda' + str(args.lmbda))
    if is_main_process(rank):
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    barrier(is_distributed)
    if args.seed is not None:
        torch.manual_seed(int(args.seed) + rank)
        random.seed(int(args.seed) + rank)

    train_dataset, test_dataset = build_datasets(args)

    device = _resolve_device(args, local_rank, is_distributed)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    import importlib
    net = importlib.import_module(f'.{args.model_name}', f'src.models').HPCM()
    if is_main_process(rank):
        print(net)
    net = net.to(device)

    lr_scheduler = lambda x : \
    1e-4 if x < 2750 else (
        3e-5 if x < 2850 else (
            1e-5 if x < 2950 else 1e-6
        )
    )

    last_epoch = 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.checkpoint:
        if is_main_process(rank):
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint, strict=True)

    barrier(is_distributed)
    if is_distributed:
        net = wrap_ddp(net, local_rank)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    writer = SummaryWriter(args.log_dir) if is_main_process(rank) else None

    best_loss = float("inf")
    global_step = 0
    try:
        for epoch in range(last_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            lr = lr_scheduler(epoch)
            for param_group in optimizer.param_groups: 
                param_group['lr'] = lr
            
            if is_main_process(rank):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            
            global_step = train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                epoch,
                global_step,
                args.clip_max_norm,
                args.model_name,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed,
            )

            if is_main_process(rank):
                loss = test_epoch(epoch, test_dataloader, net, criterion, writer, args.model_name)

                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

                if is_best:
                    print(f"epoch {epoch} is best now!")
                    torch.save(
                        unwrap_model(net).state_dict(),
                        os.path.join(args.save_path, 'epoch_' +'best' + '.pth.tar'),
                    )

                if args.save_interval > 0 and epoch % args.save_interval == 0:
                    torch.save(
                        unwrap_model(net).state_dict(),
                        os.path.join(args.save_path, 'epoch_' + str(epoch) + '.pth.tar'),
                    )
            barrier(is_distributed)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main(sys.argv[1:])
