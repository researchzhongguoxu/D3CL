# -*- coding: utf-8 -*-
"""
VisDrone truck 单类多框实验 (truck id=5, nc=1):
筛单类 -> ODS 计算 -> 三等分分组 -> 训练(Mixed 饱和 + 课程三档)

- Mixed: 训练到饱和 (epochs 上限 300, patience=20)
- N = Mixed 实际训练 epoch 数
- 课程三档: N//3, N//3+10, N//3+20
- truck 单类多框: train 3551 图/12875 框(均 3.63 框/图), val 266 图/750 框
- val 全量评估(266 张), 不做采样分离(与 Pest24 多框版口径一致)
- 全程断点续传, 供整夜 nohup 运行
"""
import os
import random
import shutil
import yaml
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

BASE = r"E:\dataset_create\visdrone"
SRC = os.path.join(BASE, "VisDrone_yolo")             # 原始 10 类数据集
TRUCK_DS = os.path.join(BASE, "VisDrone_truck_yolo")   # truck 单类 nc=1
SPLIT = os.path.join(BASE, "VisDrone_truck_split")
PROJECT = os.path.join(BASE, "runs_visdrone_truck")
VAL_YAML = os.path.join(TRUCK_DS, "data.yaml")
RES_FILE = os.path.join(BASE, "visdrone_truck_results.txt")

CLS_ID = 5
CLS_NAME = "truck"
EPSILON = 1e-6
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
MIXED_MAX_EPOCHS = 300
PATIENCE = 20
random.seed(42)


# ==================== 筛单类 ====================
def extract_single_class():
    if os.path.exists(TRUCK_DS):
        shutil.rmtree(TRUCK_DS)
    total = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        lbl_src = Path(SRC) / 'labels' / split
        img_src = Path(SRC) / 'images' / split
        dst_img = Path(TRUCK_DS) / 'images' / split
        dst_lbl = Path(TRUCK_DS) / 'labels' / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for lbl in lbl_src.glob('*.txt'):
            lines = [l for l in lbl.read_text(encoding='utf-8').splitlines() if l.strip()]
            truck_lines = []
            for l in lines:
                parts = l.split()
                if len(parts) < 5:
                    continue
                if int(float(parts[0])) != CLS_ID:
                    continue
                # 重映射 class id -> 0
                truck_lines.append("0 " + " ".join(parts[1:5]))
            if not truck_lines:
                continue
            img = None
            for ext in IMAGE_EXTS:
                p = img_src / f"{lbl.stem}{ext}"
                if p.exists():
                    img = p
                    break
            if img is None:
                continue
            shutil.copy2(str(img), str(dst_img / img.name))
            (dst_lbl / f"{lbl.stem}.txt").write_text(
                "\n".join(truck_lines) + "\n", encoding='utf-8')
            total[split] += 1
        print(f"[{split}] truck 单类图 {total[split]} 张", flush=True)

    yaml_data = {
        'path': TRUCK_DS.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'nc': 1,
        'names': [CLS_NAME],
    }
    with open(os.path.join(TRUCK_DS, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    print(f"筛单类完成 -> {TRUCK_DS} (nc=1, train={total['train']}, val={total['val']})", flush=True)


# ==================== ODS ====================
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
                x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
                pw = max(1, min(pw, w - x)); ph = max(1, min(ph, h - y))
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
        x0 = max(0, x - margin); y0 = max(0, y - margin)
        x1 = min(w, x + bw + margin); y1 = min(h, y + bh + margin)
        bg_L = L[y0:y1, x0:x1]
        if bg_L.size == 0:
            continue
        mu_bg = float(np.mean(bg_L)); sigma_bg = float(np.std(bg_L))
        C = abs(mu_obj - mu_bg) / (sigma_bg + EPSILON)
        S = float(np.sqrt(bw * bh))
        obj_gray = gray[y:y + bh, x:x + bw]
        sharpness = float(cv2.Laplacian(obj_gray, cv2.CV_64F).var()) if obj_gray.size >= 4 else 0.0
        results.append({'C': C, 'S': S, 'sharpness': sharpness, 'area': int(bw * bh)})
    return results, (h, w)


def compute_split_ods(dataset_root, split):
    img_dir = Path(dataset_root) / "images" / split
    lbl_dir = Path(dataset_root) / "labels" / split
    ods_dir = Path(dataset_root) / "ods_values" / split
    ods_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    per_image = {}
    all_sharpness = []
    for img_path in image_files:
        stem = img_path.stem
        objs, _ = compute_image_per_object(img_path, lbl_dir / f"{stem}.txt")
        if objs is None or len(objs) == 0:
            continue
        per_image[stem] = objs
        for o in objs:
            all_sharpness.append(o['sharpness'])
    if not all_sharpness:
        print(f"[{split}] 无有效目标")
        return
    def _norm(v, vmin, vmax):
        return 0.0 if abs(vmax - vmin) < EPSILON else (v - vmin) / (vmax - vmin + EPSILON)

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

    for stem, objs in per_image.items():
        ods_list = [calc_ods(o) for o in objs]
        img_ods = float(np.mean(ods_list))
        with open(ods_dir / f"{stem}.ods", 'w', encoding='utf-8') as f:
            f.write(f"{img_ods:.6f}")
    print(f"[{split}] 已计算 {len(per_image)} 张图 ODS", flush=True)


# ==================== 分组 ====================
def load_image_ods(src_root):
    ods_dir = Path(src_root) / 'ods_values' / 'train'
    items = []
    for ods_file in ods_dir.glob('*.ods'):
        try:
            ods = float(ods_file.read_text(encoding='utf-8').strip())
        except (ValueError, OSError):
            continue
        items.append((ods_file.stem, ods))
    items.sort(key=lambda x: x[1])
    return items


def copy_to_subset(src_root, stem, subset_dir):
    img_src = None
    for ext in IMAGE_EXTS:
        p = Path(src_root) / 'images' / 'train' / f"{stem}{ext}"
        if p.exists():
            img_src = p
            break
    if img_src is None:
        return
    lbl_src = Path(src_root) / 'labels' / 'train' / f"{stem}.txt"
    ods_src = Path(src_root) / 'ods_values' / 'train' / f"{stem}.ods"
    dst_img = Path(subset_dir) / 'images' / 'train'
    dst_lbl = Path(subset_dir) / 'labels' / 'train'
    dst_ods = Path(subset_dir) / 'ods_values' / 'train'
    for d in (dst_img, dst_lbl, dst_ods):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(img_src), str(dst_img / img_src.name))
    shutil.copy2(str(lbl_src), str(dst_lbl / f"{stem}.txt"))
    shutil.copy2(str(ods_src), str(dst_ods / f"{stem}.ods"))


def write_yaml(subset_dir, train_dir, val_dir, nc, names):
    yaml_data = {
        'path': str(subset_dir).replace('\\', '/'),
        'train': str(train_dir).replace('\\', '/'),
        'val': str(val_dir).replace('\\', '/'),
        'test': str(val_dir).replace('\\', '/'),
        'nc': nc,
        'names': names,
    }
    with open(Path(subset_dir) / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)


def make_split():
    val_img_dir = Path(TRUCK_DS) / 'images' / 'val'

    if os.path.exists(SPLIT):
        shutil.rmtree(SPLIT)
    os.makedirs(SPLIT, exist_ok=True)

    ordered = load_image_ods(TRUCK_DS)
    total = len(ordered)
    split1 = total // 3
    split2 = 2 * total // 3
    hard = ordered[:split1]
    medium = ordered[split1:split2]
    easy = ordered[split2:]
    print(f"总 train {total} 张: Hard {len(hard)} / Medium {len(medium)} / Easy {len(easy)}", flush=True)

    subsets = {"Hard": hard, "Medium": medium, "Easy": easy}
    for name, items in subsets.items():
        subset_dir = os.path.join(SPLIT, name)
        for stem, ods in items:
            copy_to_subset(TRUCK_DS, stem, subset_dir)
        write_yaml(subset_dir, os.path.join(subset_dir, 'images', 'train'),
                   str(val_img_dir), 1, [CLS_NAME])
        print(f"  {name}: {len(items)} 张", flush=True)

    em_dir = os.path.join(SPLIT, 'EasyMedium')
    for stem, ods in easy + medium:
        copy_to_subset(TRUCK_DS, stem, em_dir)
    write_yaml(em_dir, os.path.join(em_dir, 'images', 'train'), str(val_img_dir), 1, [CLS_NAME])
    print(f"  EasyMedium: {len(easy) + len(medium)} 张", flush=True)

    all_dir = os.path.join(SPLIT, 'All')
    for stem, ods in easy + medium + hard:
        copy_to_subset(TRUCK_DS, stem, all_dir)
    write_yaml(all_dir, os.path.join(all_dir, 'images', 'train'), str(val_img_dir), 1, [CLS_NAME])
    print(f"  All: {len(easy) + len(medium) + len(hard)} 张", flush=True)
    print(f"分组完成 -> {SPLIT}", flush=True)


# ==================== 训练 ====================
def _common(data_yaml, project, name, args, patience=None):
    d = dict(
        data=data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, project=project, name=name, exist_ok=True,
        pretrained=True, optimizer='auto', verbose=False, plots=False, val=True,
        cache='disk', amp=True, lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
        box=7.5, cls=0.5, dfl=1.5,
    )
    if patience is not None:
        d['patience'] = patience
    return d


def _read_last_epoch(csv_path):
    try:
        with open(csv_path) as f:
            lines = [ln for ln in f if ln.strip()]
        if len(lines) >= 2:
            return int(float(lines[-1].split(',')[0]))
    except Exception:
        pass
    return 0


def _train_one(model_path, data_yaml, epochs, name, project, args):
    last_pt = os.path.join(project, name, 'weights', 'last.pt')
    best_pt = os.path.join(project, name, 'weights', 'best.pt')
    csv_path = os.path.join(project, name, 'results.csv')
    if _read_last_epoch(csv_path) >= epochs:
        print(f"  [skip] {name} 已完成 {epochs}ep", flush=True)
        return best_pt
    common = _common(data_yaml, project, name, args)
    if os.path.exists(last_pt):
        print(f"  [resume] {name} 续训至 {epochs}ep", flush=True)
        model = YOLO(last_pt)
        model.train(epochs=epochs, resume=True, **common)
    else:
        print(f"  [start] {name} 从头训 {epochs}ep", flush=True)
        model = YOLO(model_path)
        model.train(epochs=epochs, **common)
    return best_pt


def train_mixed(args):
    os.makedirs(PROJECT, exist_ok=True)
    flag = os.path.join(PROJECT, 'mixed_saturated.txt')
    best_pt = os.path.join(PROJECT, 'mixed', 'weights', 'best.pt')
    if os.path.exists(flag):
        N = int(open(flag).read().strip())
        print(f"  Mixed 已饱和训练完成 (N={N})", flush=True)
        return N, best_pt
    all_yaml = os.path.join(SPLIT, 'All', 'data.yaml')
    last_pt = os.path.join(PROJECT, 'mixed', 'weights', 'last.pt')
    common = _common(all_yaml, PROJECT, 'mixed', args, patience=PATIENCE)
    if os.path.exists(last_pt):
        print("  [resume] Mixed 续训至饱和", flush=True)
        model = YOLO(last_pt)
        model.train(epochs=MIXED_MAX_EPOCHS, resume=True, **common)
    else:
        print(f"  [start] Mixed 训至饱和 (上限 {MIXED_MAX_EPOCHS}, patience={PATIENCE})", flush=True)
        model = YOLO('yolov8n.pt')
        model.train(epochs=MIXED_MAX_EPOCHS, **common)
    N = _read_last_epoch(os.path.join(PROJECT, 'mixed', 'results.csv'))
    with open(flag, 'w') as f:
        f.write(str(N))
    print(f"  Mixed 饱和完成 N={N}", flush=True)
    return N, best_pt


def evaluate(model_path, label, args):
    model = YOLO(model_path)
    r = model.val(data=VAL_YAML, batch=args.batch, imgsz=args.imgsz,
                  device=args.device, workers=args.workers, verbose=False)
    m = r.box
    return {'label': label, 'mAP50': round(float(m.map50), 4),
            'mAP50-95': round(float(m.map), 4),
            'Precision': round(float(np.mean(m.p)), 4),
            'Recall': round(float(np.mean(m.r)), 4)}


def run_curriculum(budget, args):
    easy_yaml = os.path.join(SPLIT, 'Easy', 'data.yaml')
    em_yaml = os.path.join(SPLIT, 'EasyMedium', 'data.yaml')
    all_yaml = os.path.join(SPLIT, 'All', 'data.yaml')
    tag = f"b{budget}"
    print(f"\n  [课程] 预算 {budget}ep/阶段", flush=True)
    b1 = _train_one('yolov8n.pt', easy_yaml, budget, f"{tag}_stage1_easy", PROJECT, args)
    b2 = _train_one(b1, em_yaml, budget, f"{tag}_stage2_em", PROJECT, args)
    b3 = _train_one(b2, all_yaml, budget, f"{tag}_stage3_all", PROJECT, args)
    return evaluate(b3, f"{budget}ep Stage3", args)


def summarize(N, budgets, mixed, s3_list):
    lines = ["=" * 70,
             f"VisDrone truck 单类实验  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 70,
             f"Mixed 饱和实际 N={N} (patience={PATIENCE})",
             f"课程档位: " + " / ".join(f"{b}ep" for b in budgets),
             f"{'配置':<20}{'mAP50':>10}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}",
             "-" * 70,
             f"{mixed['label']:<20}{mixed['mAP50']:>10}{mixed['mAP50-95']:>12}"
             f"{mixed['Precision']:>12}{mixed['Recall']:>10}"]
    for s3 in s3_list:
        lines.append(f"{s3['label']:<20}{s3['mAP50']:>10}{s3['mAP50-95']:>12}"
                     f"{s3['Precision']:>12}{s3['Recall']:>10}")
    lines.append("\n增益 (Stage3 - Mixed):")
    for b, s3 in zip(budgets, s3_list):
        lines.append(f"  {b}ep: {s3['mAP50'] - mixed['mAP50']:+.4f}")
    with open(RES_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='all',
                        choices=['extract', 'ods', 'split', 'train', 'all'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    if args.stage in ('extract', 'all'):
        print("== 筛单类 (truck id=5 -> nc=1) ==", flush=True)
        extract_single_class()
    if args.stage in ('ods', 'all'):
        print("== ODS 计算 ==", flush=True)
        compute_split_ods(TRUCK_DS, 'train')
        compute_split_ods(TRUCK_DS, 'val')
    if args.stage in ('split', 'all'):
        print("== 分组 (三等分) ==", flush=True)
        make_split()
    if args.stage in ('train', 'all'):
        print("== 训练 ==", flush=True)
        N, mixed_best = train_mixed(args)
        mixed = evaluate(mixed_best, 'Mixed', args)
        print(f"  Mixed -> {mixed}", flush=True)
        budgets = [N // 3, N // 3 + 10, N // 3 + 20, N // 3 + 30]
        print(f"  档位预算: {budgets}", flush=True)
        s3_list = []
        for b in budgets:
            s3 = run_curriculum(b, args)
            s3_list.append(s3)
            print(f"  {b}ep Stage3 -> {s3}", flush=True)
        summarize(N, budgets, mixed, s3_list)


if __name__ == '__main__':
    main()
