"""Batch binary Canny export (OpenCV). Training uses ldm.canny_util."""

import cv2
import numpy as np
from pathlib import Path

from ldm.canny_util import compute_binary_canny_bgr

def count_neighbors(mask, y, x):
    patch = mask[max(0, y-1):y+2, max(0, x-1):x+2]
    return int(np.count_nonzero(patch) - 1)

def extract_closed_edges_from_canny(canny, min_size=30, do_morph_close=True):

    if do_morph_close:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    edge01 = (edges > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge01, connectivity=8)

    closed_map = np.zeros_like(edge01, dtype=np.uint8)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_size:
            continue

        mask = (labels == lbl).astype(np.uint8)
        ys, xs = np.where(mask)

        endpoints = 0
        for y, x in zip(ys, xs):
            if count_neighbors(mask, y, x) == 1:
                endpoints += 1

        if endpoints == 0:
            closed_map[mask > 0] = 255

    return closed_map

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    file_root = Path("/WAVE/users2/unix/zdong/vipteam/Calvin/dataset/DIV2K/DIV2K_valid_HR_512_512")
    edges_root = Path("/WAVE/users2/unix/zdong/vipteam/Calvin/dataset/DIV2K/DIV2K_valid_HR_512_512_canny")
    # closed_edges_root = Path("/WAVE/users2/unix/zdong/vipteam/Calvin/dataset/DIV2K_train_HR_1280_720_closed_canny_edges")

    image_paths = sorted(
        p
        for p in file_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Found {len(image_paths)} images under {file_root}")

    for i, file in enumerate(image_paths, 1):
        rel_path = file.relative_to(file_root)
        print(f"[{i}/{len(image_paths)}] {rel_path}")
        img = cv2.imread(str(file))
        if img is None:
            print(f"  skip (could not read): {file}")
            continue
        canny = compute_binary_canny_bgr(img)

        print("density: ", np.sum(canny) / (canny.shape[0] * canny.shape[1] * 255))

        save_path = edges_root / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canny)

        # closed_edges = extract_closed_edges_from_canny(canny)

        # save_path = closed_edges_root / rel_path
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(save_path, closed_edges)

if __name__ == "__main__":
    main()