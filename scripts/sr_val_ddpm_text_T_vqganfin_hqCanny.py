"""
Inference for StableSR + HQ Canny struct condition (paired LQ + GT).

- LQ: --init-img  (blind SR input, same as original script)
- GT: --gt-img    (only used to build HQ Canny edges -> z_canny, and optional metrics)
- Does NOT modify the original sr_val_ddpm_text_T_vqganfin_old.py

Canny: OpenCV binary edges from GT (canny.py / training). ControlNet uses RGB hint; legacy uses VAE latent.
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import PIL
import torch
import torchvision
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from tqdm import trange

from ldm.util import instantiate_from_config
from scripts.wavelet_color_fix import adaptive_instance_normalization, wavelet_reconstruction

# Reuse helpers from the original inference script
from scripts.sr_val_ddpm_text_T_vqganfin_old import (
    load_model_from_config,
    load_img,
    space_timesteps,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder):
    names = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS:
            names.append(f)
    return names


def load_img_tensor(path, size):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    t = torch.from_numpy(arr[None].transpose(0, 3, 1, 2))
    t = 2.0 * t - 1.0
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
    ])
    t = transform(t)
    return t.clamp(-1, 1)


def load_canny_hint_tensor(path, size, device):
    """Load pre-compressed/decoded canny PNG as [1,3,H,W] float in [0,1]."""
    image = Image.open(path).convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    t = torch.from_numpy(arr[None].transpose(0, 3, 1, 2))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
    ])
    return transform(t).clamp(0, 1).to(device)


def resolve_canny_path(canny_dir, fname):
    canny_path = os.path.join(canny_dir, fname)
    if os.path.exists(canny_path):
        return canny_path
    base = os.path.splitext(fname)[0]
    for ext in IMAGE_EXTENSIONS:
        alt = os.path.join(canny_dir, base + ext)
        if os.path.exists(alt):
            return alt
    return None


def encode_hq_canny_hint(model, gt_minus1_1, device):
    """Match training: binary OpenCV Canny, 3ch in [0,1] for ControlNet."""
    from ldm.canny_util import compute_binary_canny_tensor

    gt_01 = (gt_minus1_1 + 1.0) * 0.5
    return compute_binary_canny_tensor(gt_01).to(device)


def encode_hq_canny_latent(model, gt_minus1_1, device):
    """Legacy path: VAE latent of canny RGB."""
    canny_hint = encode_hq_canny_hint(model, gt_minus1_1, device)
    canny_rgb = canny_hint * 2.0 - 1.0
    enc = model.encode_first_stage(canny_rgb)
    return model.get_first_stage_encoding(enc)


def compute_metrics(out_dir, gt_dir, names):
    from scripts.util_image import calculate_ms_ssim, calculate_psnr, calculate_ssim

    psnrs, ssims, mssims, miss = [], [], [], 0
    for f in names:
        gt_path = os.path.join(gt_dir, f)
        out_path = os.path.join(out_dir, os.path.splitext(f)[0] + ".png")
        if not os.path.exists(gt_path) or not os.path.exists(out_path):
            miss += 1
            continue
        gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32)
        out = np.array(Image.open(out_path).convert("RGB")).astype(np.float32)
        if gt.shape != out.shape:
            out_img = Image.fromarray(out.astype(np.uint8))
            out_img = out_img.resize((gt.shape[1], gt.shape[0]), resample=PIL.Image.LANCZOS)
            out = np.array(out_img).astype(np.float32)
        psnrs.append(calculate_psnr(out, gt, border=0, ycbcr=True))
        ssims.append(calculate_ssim(out, gt, border=0, ycbcr=True))
        mssims.append(calculate_ms_ssim(out, gt, border=0, ycbcr=True))
    if len(psnrs) == 0:
        return None
    return {
        "matched": len(psnrs),
        "missing": miss,
        "psnr_y_mean": float(np.mean(psnrs)),
        "ssim_y_mean": float(np.mean(ssims)),
        "ms_ssim_y_mean": float(np.mean(mssims)),
    }


def main():
    parser = argparse.ArgumentParser(description="StableSR inference with HQ Canny (LQ + GT canny)")
    parser.add_argument("--init-img", type=str, required=True, help="LQ image folder")
    parser.add_argument("--gt-img", type=str, required=True, help="GT folder (canny + metrics)")
    parser.add_argument("--outdir", type=str, default="outputs/hqCanny_eval")
    parser.add_argument("--ddpm_steps", type=int, default=200)
    parser.add_argument("--config", type=str, default="configs/stableSRNew/v2-finetune_text_T_512_hqCanny.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="finetuned hqCanny checkpoint")
    parser.add_argument("--vqgan_ckpt", type=str, default="checkpoints/vqgan_cfw_00011.ckpt")
    parser.add_argument(
        "--vqgan-config",
        type=str,
        default="configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml",
        help="use autoencoder_kl_64x64x4_resi_canny.yaml after training CFW+Canny",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1, help="batch size")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--dec_w", type=float, default=0.5)
    parser.add_argument("--colorfix_type", type=str, default="adain", choices=["adain", "wavelet", "nofix"])
    parser.add_argument("--precision", type=str, default="autocast", choices=["full", "autocast"])
    parser.add_argument("--compute_metrics", action="store_true", help="PSNR/SSIM/MS-SSIM vs GT after inference")
    parser.add_argument("--save_canny_vis", action="store_true", help="save canny preview png next to outputs")
    parser.add_argument(
        "--canny-dir",
        type=str,
        default="",
        help="Use pre-computed canny maps (e.g. HPCM decoded) instead of GT-derived edges",
    )
    parser.add_argument(
        "--zero-canny-hint",
        action="store_true",
        help="ablation: use all-zero Canny hint (ControlNet still runs; hint is black)",
    )
    parser.add_argument(
        "--no-canny",
        action="store_true",
        help="ablation: disable ControlNet branch entirely (same as canny_hint=None)",
    )
    parser.add_argument(
        "--canny-cond-weight",
        type=float,
        default=None,
        help="override yaml canny_cond_weight (0 = zero control strength after ControlNet)",
    )
    opt = parser.parse_args()
    if opt.zero_canny_hint and opt.no_canny:
        parser.error("Use only one of --zero-canny-hint or --no-canny")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(opt.seed)
    os.makedirs(opt.outdir, exist_ok=True)

    vqgan_config = OmegaConf.load(opt.vqgan_config)
    if getattr(vqgan_config.model.params, "use_canny_cfw", False):
        print("VQGAN: use_canny_cfw=True (LQ + Canny features in CFW fusion)")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt).to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    config = OmegaConf.load(opt.config)
    # Finetuned --ckpt already has canny_controlnet.*; do not preload .safetensors.
    if OmegaConf.select(config, "model.params.controlnet_ckpt_path"):
        print("Inference: skip controlnet_ckpt_path (use weights from --ckpt).")
        config.model.params.controlnet_ckpt_path = None
    model = load_model_from_config(config, opt.ckpt).to(device)
    if opt.canny_cond_weight is not None:
        model.canny_cond_weight = float(opt.canny_cond_weight)
        print(f"Override canny_cond_weight = {model.canny_cond_weight}")
    if not getattr(model, "use_hq_canny_cond", False):
        print("WARNING: config.model.params.use_hq_canny_cond is False; canny branch may be ignored.")

    use_controlnet = getattr(model, "canny_controlnet", None) is not None and not opt.no_canny
    if opt.no_canny:
        print("Ablation: --no-canny (ControlNet disabled)")
    elif opt.zero_canny_hint:
        print("Ablation: --zero-canny-hint (hint = all zeros, ControlNet still runs)")
    elif opt.canny_dir:
        print(f"Canny: pre-computed from {opt.canny_dir}, canny_cond_weight={model.canny_cond_weight}")
    else:
        print(f"Canny: GT edges, canny_cond_weight={model.canny_cond_weight}")

    model.register_schedule(
        given_betas=None, beta_schedule="linear", timesteps=1000,
        linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3,
    )
    model.num_timesteps = 1000
    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = sorted(list(use_timesteps))
    # register_schedule recreates buffers on CPU; move them back to GPU (same as old script)
    model = model.to(device)

    lq_names = list_images(opt.init_img)
    pairs = []
    for f in lq_names:
        gt_path = os.path.join(opt.gt_img, f)
        if not os.path.exists(gt_path):
            base = os.path.splitext(f)[0]
            for ext in IMAGE_EXTENSIONS:
                alt = os.path.join(opt.gt_img, base + ext)
                if os.path.exists(alt):
                    gt_path = alt
                    break
        if not os.path.exists(gt_path):
            print(f"skip (no GT): {f}")
            continue
        if opt.canny_dir and not opt.no_canny and not opt.zero_canny_hint:
            canny_path = resolve_canny_path(opt.canny_dir, f)
            if canny_path is None:
                print(f"skip (no canny): {f}")
                continue
        out_png = os.path.join(opt.outdir, os.path.splitext(f)[0] + ".png")
        if os.path.exists(out_png):
            continue
        pairs.append((f, os.path.join(opt.init_img, f), gt_path))

    print(f"HQ Canny inference: {len(pairs)} pairs (LQ + "
          f"{'no canny' if opt.no_canny else 'zero canny' if opt.zero_canny_hint else 'canny-dir' if opt.canny_dir else 'GT canny'})")
    if len(pairs) == 0:
        print("Nothing to run.")
        if opt.compute_metrics:
            m = compute_metrics(opt.outdir, opt.gt_img, lq_names)
            if m:
                print(m)
        return

    batch_size = opt.n_samples
    niters = math.ceil(len(pairs) / batch_size)
    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    processed_names = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                idx = 0
                for n in trange(niters, desc="Sampling"):
                    batch_pairs = pairs[idx: idx + batch_size]
                    idx += batch_size
                    if not batch_pairs:
                        break

                    lq_list, gt_list = [], []
                    for _, lq_p, gt_p in batch_pairs:
                        lq_list.append(load_img_tensor(lq_p, opt.input_size))
                        gt_list.append(load_img_tensor(gt_p, opt.input_size))
                    init_image = torch.cat(lq_list, dim=0).to(device)
                    gt_image = torch.cat(gt_list, dim=0).to(device)

                    init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
                    init_latent = model.get_first_stage_encoding(init_latent_generator)
                    canny_hint = None
                    z_canny = None
                    if use_controlnet:
                        if opt.zero_canny_hint:
                            canny_hint = encode_hq_canny_hint(model, gt_image, device)
                            canny_hint = torch.zeros_like(canny_hint)
                        elif opt.canny_dir:
                            hints = []
                            for fname, _, _ in batch_pairs:
                                cp = resolve_canny_path(opt.canny_dir, fname)
                                hints.append(load_canny_hint_tensor(cp, opt.input_size, device))
                            canny_hint = torch.cat(hints, dim=0)
                        else:
                            canny_hint = encode_hq_canny_hint(model, gt_image, device)
                    elif getattr(model, "canny_structcond_stage_model", None) is not None:
                        z_canny = encode_hq_canny_latent(model, gt_image, device)

                    text_init = [""] * init_image.size(0)
                    semantic_c = model.cond_stage_model(text_init)

                    noise = torch.randn_like(init_latent)
                    t = repeat(torch.tensor([999]), "1 -> b", b=init_image.size(0)).to(device).long()
                    x_T = model.q_sample_respace(
                        x_start=init_latent, t=t,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        noise=noise,
                    )
                    x_T = None

                    samples, _ = model.sample(
                        cond=semantic_c,
                        struct_cond=init_latent,
                        z_canny=z_canny,
                        canny_hint=canny_hint if use_controlnet else None,
                        batch_size=init_image.size(0),
                        timesteps=opt.ddpm_steps,
                        time_replace=opt.ddpm_steps,
                        x_T=x_T,
                        return_intermediates=True,
                    )
                    enc_fea_canny = None
                    if (
                        getattr(vq_model.decoder, "use_canny_cfw", False)
                        and model.canny_controlnet is not None
                        and canny_hint is not None
                    ):
                        from ldm.canny_util import encode_hint_for_cfw
                        enc_fea_canny = encode_hint_for_cfw(
                            model.canny_controlnet, canny_hint, t, semantic_c
                        )
                    x_samples = vq_model.decode(
                        samples * 1.0 / model.scale_factor, enc_fea_lq, enc_fea_canny=enc_fea_canny
                    )
                    if opt.colorfix_type == "adain":
                        x_samples = adaptive_instance_normalization(x_samples, init_image)
                    elif opt.colorfix_type == "wavelet":
                        x_samples = wavelet_reconstruction(x_samples, init_image)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if opt.save_canny_vis:
                        canny_vis = canny_hint

                    for i, (fname, _, _) in enumerate(batch_pairs):
                        basename = os.path.splitext(os.path.basename(fname))[0]
                        x_sample = 255.0 * rearrange(x_samples[i].cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(opt.outdir, basename + ".png")
                        )
                        if opt.save_canny_vis:
                            cv = 255.0 * rearrange(canny_vis[i].cpu().numpy(), "c h w -> h w c")
                            Image.fromarray(cv.astype(np.uint8)).save(
                                os.path.join(opt.outdir, basename + "_canny.png")
                            )
                        processed_names.append(fname)

                print(f"Done in {time.time() - tic:.1f}s -> {opt.outdir}")

    if opt.compute_metrics:
        m = compute_metrics(opt.outdir, opt.gt_img, lq_names)
        if m:
            print("Metrics (YCbCr Y):")
            print(f"  matched: {m['matched']}, missing: {m['missing']}")
            print(f"  PSNR(Y) mean:    {m['psnr_y_mean']:.4f}")
            print(f"  SSIM(Y) mean:    {m['ssim_y_mean']:.4f}")
            print(f"  MS-SSIM(Y) mean: {m['ms_ssim_y_mean']:.4f}")
        else:
            print("No metric pairs found.")


if __name__ == "__main__":
    main()
