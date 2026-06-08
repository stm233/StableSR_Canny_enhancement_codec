"""Binary Canny edges (OpenCV), matching StableSR/canny.py."""

import cv2
import numpy as np
import torch


def compute_binary_canny_bgr(img_bgr, low=100, high=200, blur_ksize=5, blur_sigma=1.4):
    """img_bgr: uint8 HWC BGR. Returns uint8 HxW {0, 255}."""
    lur = cv2.GaussianBlur(img_bgr, (blur_ksize, blur_ksize), blur_sigma)
    return cv2.Canny(lur, low, high)


def compute_binary_canny_tensor(gt_01, low=100, high=200):
    """
    gt_01: [B,3,H,W] float in [0,1] RGB.
    Returns [B,3,H,W] float in [0,1] (3-channel binary edge map for ControlNet hint).
    """
    device = gt_01.device
    b, _, h, w = gt_01.shape
    out = []
    gt_np = (gt_01.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(b):
        rgb = np.transpose(gt_np[i], (1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        edge = compute_binary_canny_bgr(bgr)
        edge3 = np.stack([edge, edge, edge], axis=0).astype(np.float32) / 255.0
        out.append(torch.from_numpy(edge3))
    return torch.stack(out, dim=0).to(device)


def encode_hint_for_cfw(controlnet, hint_01, timesteps, context):
    """CFW branch: features from ControlNet.input_hint_block (shared with UNet control path)."""
    return controlnet.encode_hint_for_cfw(hint_01, timesteps, context)
