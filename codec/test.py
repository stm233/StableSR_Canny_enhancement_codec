import csv
import json
import math
import glob
import os
import time
from datetime import datetime
import torch
import argparse
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from pytorch_msssim import ms_ssim, ssim

# Default ms_ssim uses 5 scales + win_size=11 -> min side must be > 160.
_MS_SSIM_LEVELS = 5
_MS_SSIM_DEFAULT_WIN = 11
       
def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )
    
def load_image(filepath: str):
    return Image.open(filepath).convert("RGB")

def img2torch(img: Image.Image):
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    x = x.squeeze(0).detach().cpu().clamp(0, 1).mul(255).round().byte()
    return Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255):
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255):
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    return metrics


def _ms_ssim_min_side(win_size: int, levels: int = _MS_SSIM_LEVELS) -> int:
    return (win_size - 1) * (2 ** (levels - 1))


def compute_msssim_db(
    x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0
) -> tuple:
    """MS-SSIM in dB; auto-fallback for small images (e.g. 64x64)."""
    min_side = min(x.shape[-2], x.shape[-1])
    metric = "ms-ssim"
    win_size = _MS_SSIM_DEFAULT_WIN

    if min_side <= _ms_ssim_min_side(win_size):
        for ws in (7, 5, 3):
            if min_side > _ms_ssim_min_side(ws):
                win_size = ws
                metric = f"ms-ssim (win={ws})"
                break
        else:
            val = ssim(x_hat, x, data_range=data_range)
            metric = "ssim"
            val = float(val.clamp(max=1 - 1e-8))
            return 10 * math.log10(1 / (1 - val)), metric

    val = ms_ssim(x_hat, x, data_range=data_range, win_size=win_size)
    val = float(val.clamp(max=1 - 1e-8))
    return 10 * math.log10(1 / (1 - val)), metric

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        if type(val) == torch.Tensor:
            val = val.detach().cpu()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def get_scale_table(min, max, levels):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _results_dir(args) -> Optional[str]:
    path = args.results_dir or args.outdir
    return path or None


def _ckpt_tag(ckpt: str) -> str:
    return os.path.splitext(os.path.basename(ckpt))[0]


def _save_results(
    results_dir: str,
    args,
    ckpt: str,
    per_image: List[Dict[str, Any]],
    summary: Dict[str, Any],
):
    if len(args.checkpoint) > 1:
        results_dir = os.path.join(results_dir, _ckpt_tag(ckpt))
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "per_image.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "psnr", "msssim_db", "msssim_metric", "bpp", "y_bpp",
                "z_bpp", "enc_time", "dec_time",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "checkpoint": ckpt,
        "dataset": args.dataset,
        "device": args.device,
        "num_images": len(per_image),
        "outdir": args.outdir or None,
        "summary": summary,
        "per_image": per_image,
    }
    json_path = os.path.join(results_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(results_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"timestamp: {payload['timestamp']}\n")
        f.write(f"model: {args.model_name}\n")
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"num_images: {len(per_image)}\n")
        f.write("\n[Average]\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n[Per image]\n")
        for row in per_image:
            f.write(
                f"{row['image']}\t"
                f"PSNR={row['psnr']:.4f}\t"
                f"MS-SSIM={row['msssim_db']:.4f}\t"
                f"bpp={row['bpp']:.6f}\t"
                f"enc={row['enc_time']:.4f}s\t"
                f"dec={row['dec_time']:.4f}s\n"
            )

    print(f"Results saved to: {results_dir}")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")


def test(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Saving reconstructions to: {args.outdir}")
    results_root = _results_dir(args)
    if results_root:
        os.makedirs(results_root, exist_ok=True)
        print(f"Saving metrics to: {results_root}")
    ##### dataset
    images_list = glob.glob(f'{args.dataset}/*.png')
    if not images_list:
        raise FileNotFoundError(f"No .png images found in: {args.dataset}")

    sample = load_image(images_list[0])
    sample_side = min(sample.size)
    if sample_side <= _ms_ssim_min_side(_MS_SSIM_DEFAULT_WIN):
        print(
            f"Note: image short side={sample_side}px < "
            f"{_ms_ssim_min_side(_MS_SSIM_DEFAULT_WIN) + 1}px; "
            "MS-SSIM will use reduced window or SSIM fallback."
        )

    ##### load model
    import importlib
    net = importlib.import_module(f'.{args.model_name}', f'src.models').HPCM
        
    args.checkpoint = [args.checkpoint]
    # suggest:
    # args.checkpoint = [
    #     '/path-to-ckpt/0.0018.pth.tar', 
    #     '/path-to-ckpt/0.0035.pth.tar', 
    #     '/path-to-ckpt/0.0067.pth.tar', 
    #     '/path-to-ckpt/0.013.pth.tar', 
    #     '/path-to-ckpt/0.025.pth.tar', 
    #     '/path-to-ckpt/0.0483.pth.tar', 
    # ]
    bpp_all = []
    psnr_all = []
    ssim_all = []
    for ckpt in args.checkpoint:
        print("Loading", ckpt)
        checkpoint = torch.load(ckpt, map_location=device)
        model = net()
        model.eval()
        model.load_state_dict(checkpoint, strict=True)
        model.update(get_scale_table(0.12, 64, args.num))
        model = model.to(device)

        bpp_loss = AverageMeter()
        psnr = AverageMeter()
        ssim = AverageMeter()
        y_bpp = AverageMeter()
        z_bpp = AverageMeter()
        enc_time = AverageMeter()
        dec_time = AverageMeter()
        per_image: List[Dict[str, Any]] = []

        for img_path in sorted(images_list):
            
            img = load_image(img_path)
            x = img2torch(img)
            h, w = x.size(2), x.size(3)
            x = x.to(device)
            p = 256
            x_pad = pad(x, p)
            img_name = img_path.split('/')[-1]
            print(img_name)
            _sync(device)
            enc_start = time.time()
            with torch.no_grad():
                out_enc = model.compress(x_pad)
            _sync(device)
            enc_t = time.time() - enc_start

            _sync(device)
            dec_start = time.time()
            with torch.no_grad():
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            _sync(device)
            dec_t = time.time() - dec_start
            x_hat = crop(out_dec["x_hat"], (h,w))

            if args.outdir:
                torch2img(x_hat).save(os.path.join(args.outdir, img_name))

            psnr_img = compute_metrics(x, x_hat, 255)['psnr']

            msssim_db, msssim_metric = compute_msssim_db(x_hat, x, data_range=1.0)

            num_pixels = h*w
            bpp_img = sum(len(s) for s in out_enc["strings"]) * 8.0 / num_pixels
            ybpp_img = len(out_enc["strings"][0]) * 8.0 / num_pixels
            zbpp_img = len(out_enc["strings"][1]) * 8.0 / num_pixels

            print('image name:',img_name)
            print(
                f"{img_name}"
                f"\tPSNR: {psnr_img} |"
                f"\t{msssim_metric}: {msssim_db} |"
                f"\tBpp loss: {bpp_img} |"
                f"\ty bpp: {ybpp_img} |"
                f"\tz bpp: {zbpp_img} |"
                f"\tenc time: {enc_t} |"
                f"\tdec time: {dec_t} |"
            )

            bpp_loss.update(bpp_img)
            psnr.update(psnr_img)
            ssim.update(msssim_db)
            y_bpp.update(ybpp_img)
            z_bpp.update(zbpp_img)
            enc_time.update(enc_t)
            dec_time.update(dec_t)
            per_image.append({
                "image": img_name,
                "psnr": float(psnr_img),
                "msssim_db": float(msssim_db),
                "msssim_metric": msssim_metric,
                "bpp": float(bpp_img),
                "y_bpp": float(ybpp_img),
                "z_bpp": float(zbpp_img),
                "enc_time": float(enc_t),
                "dec_time": float(dec_t),
            })

        summary = {
            "psnr": float(psnr.avg),
            "msssim_db": float(ssim.avg),
            "msssim_metric": per_image[0]["msssim_metric"] if per_image else "",
            "bpp": float(bpp_loss.avg),
            "y_bpp": float(y_bpp.avg),
            "z_bpp": float(z_bpp.avg),
            "enc_time": float(enc_time.avg),
            "dec_time": float(dec_time.avg),
        }
        print(
            f"Test:"
            f"\tPSNR: {summary['psnr']} |"
            f"\t{summary.get('msssim_metric', 'MS-SSIM')}: {summary['msssim_db']} |"
            f"\tBpp loss: {summary['bpp']} |"
            f"\ty bpp: {summary['y_bpp']} |"
            f"\tz bpp: {summary['z_bpp']} |"
            f"\tenc time: {summary['enc_time']} |"
            f"\tdec time: {summary['dec_time']} |"
        )
        if results_root:
            _save_results(results_root, args, ckpt, per_image, summary)
        bpp_all.append(bpp_loss.avg)
        psnr_all.append(psnr.avg)
        ssim_all.append(ssim.avg)
    print(bpp_all)
    print(psnr_all)
    print(ssim_all)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--model_name", type=str, default="HPCM_Base")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("-num", "--num", type=int, default=60)
    parser.add_argument("-data", "--dataset", type=str, default='')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (use cpu when GPU is busy)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="If set, save reconstructed images here (same filenames as input)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Save per-image CSV + results.json + summary.txt here "
             "(defaults to --outdir if not set)",
    )
    args = parser.parse_args()
    print(args)
    test(args)
