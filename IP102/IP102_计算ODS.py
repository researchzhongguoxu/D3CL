# -*- coding: utf-8 -*-
"""IP102 Cicadellidae ODS 计算 (与 Pest24/VisDrone 同款)
C = |mu_obj - mu_bg| / (sigma_bg + eps)  -- CIELAB L 通道
S = sqrt(w*h)
B = 1 - minmax_norm(Laplacian方差)
C、S、B 全部 min-max 归一化后再算 ODS (论文式 1-4): ODS = C_tilde * S_tilde / (1 + B_tilde)
图级 ODS = 图内所有框 ODS 均值 (单类单框, 即框级)
"""
import os
import cv2
import numpy as np
from pathlib import Path

EPSILON = 1e-6


def compute_image_per_object(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = []
    if lbl_path is not None and Path(lbl_path).exists():
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(p) for p in parts[:5])
                x = int(round((cx - bw / 2.0) * w))
                y = int(round((cy - bh / 2.0) * h))
                pw = int(round(bw * w))
                ph = int(round(bh * h))
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                pw = max(1, min(pw, w - x))
                ph = max(1, min(ph, h - y))
                boxes.append((x, y, pw, ph))

    if not boxes:
        return None, (h, w)

    results = []
    for (x, y, bw, bh) in boxes:
        obj_L = L[y:y + bh, x:x + bw]
        if obj_L.size == 0:
            continue
        mu_obj = float(np.mean(obj_L))

        margin = int(max(bw, bh) * 0.2)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w, x + bw + margin)
        y1 = min(h, y + bh + margin)
        bg_L = L[y0:y1, x0:x1]
        if bg_L.size == 0:
            continue
        mu_bg = float(np.mean(bg_L))
        sigma_bg = float(np.std(bg_L))

        C = abs(mu_obj - mu_bg) / (sigma_bg + EPSILON)
        S = float(np.sqrt(bw * bh))

        obj_gray = gray[y:y + bh, x:x + bw]
        if obj_gray.size < 4:
            sharpness = 0.0
        else:
            sharpness = float(cv2.Laplacian(obj_gray, cv2.CV_64F).var())

        results.append({'C': C, 'S': S, 'sharpness': sharpness, 'area': int(bw * bh)})

    return results, (h, w)


def compute_split_ods(dataset_root, split):
    ods_dir_name = 'ods_values'
    img_dir = Path(dataset_root) / "images" / split
    lbl_dir = Path(dataset_root) / "labels" / split
    ods_dir = Path(dataset_root) / ods_dir_name / split
    ods_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in img_dir.iterdir()
                          if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')])

    per_image = {}
    all_sharpness = []
    for img_path in image_files:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        objs, size = compute_image_per_object(img_path, lbl_path)
        if objs is None or len(objs) == 0:
            continue
        per_image[stem] = objs
        for o in objs:
            all_sharpness.append(o['sharpness'])

    if not all_sharpness:
        print(f"[{split}] 无有效目标, 终止")
        return

    def _norm(v, vmin, vmax):
        if abs(vmax - vmin) < EPSILON:
            return 0.0
        return (v - vmin) / (vmax - vmin + EPSILON)

    # 论文式 1-4：C、S、B 全部 min-max 归一化后再算 ODS
    all_C = [o['C'] for o in sum(per_image.values(), [])]
    all_S = [o['S'] for o in sum(per_image.values(), [])]
    C_min, C_max = min(all_C), max(all_C)
    S_min, S_max = min(all_S), max(all_S)
    L_min, L_max = min(all_sharpness), max(all_sharpness)
    B_all = [1.0 - _norm(L, L_min, L_max) for L in all_sharpness]
    B_min, B_max = min(B_all), max(B_all)

    def calc_ods(o):
        C_tilde = _norm(o['C'], C_min, C_max)
        S_tilde = _norm(o['S'], S_min, S_max)
        B = 1.0 - _norm(o['sharpness'], L_min, L_max)
        B_tilde = _norm(B, B_min, B_max)
        return (C_tilde * S_tilde) / (1.0 + B_tilde)

    print(f"[{split}] sharpness:[{L_min:.2f},{L_max:.2f}] C:[{C_min:.4f},{C_max:.4f}] S:[{S_min:.2f},{S_max:.2f}]")

    for stem, objs in per_image.items():
        ods_list = [calc_ods(o) for o in objs]
        img_ods = float(np.mean(ods_list))
        with open(ods_dir / f"{stem}.ods", 'w', encoding='utf-8') as f:
            f.write(f"{img_ods:.6f}")

    print(f"[{split}] 已计算 {len(per_image)} 张图 ODS")


if __name__ == "__main__":
    DATASET_ROOT = r"E:\dataset_create\IP102_Cicadellidae\IP102_yolo"
    compute_split_ods(DATASET_ROOT, "train")
    compute_split_ods(DATASET_ROOT, "val")
    print("ODS 计算完成")
