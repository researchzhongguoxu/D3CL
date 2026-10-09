# -*- coding: utf-8 -*-
"""
COCO person 单类 · ODS 课程学习实验 (person nc=1, 真实数据):
(已预分层采样 2000 张) -> ODS -> 三等分分组 -> 训练(Mixed 饱和 + 课程三档)

目的: 得到"课程学习 Stage3 > Mixed"的正增益结果 (中小目标为主, 纯真实数据)。
机制: COCO person 的 S=sqrt(area) 跨度 2.9~636px (约 220 倍, 远超 VisDrone pedestrian 的 100 倍),
      small 40.3% + medium 33.6% = 73.9% 中小目标, large 26%, 符合"中小目标为主、可有大目标"要求,
      ODS 三档难度差异更大, 更可能复现论文合成数据的正增益。

- Mixed: 训练到饱和 (epochs 上限 300, patience=20)
- N_mixed = Mixed 实际训练 epoch 数
- 课程三档: N_mixed//3, N_mixed//3+10, N_mixed//3+20
- 数据: coco_person_yolo (train 2000 张已按 S 分层采样覆盖全谱系, val 2693 张全量)
- 训练 seed: --seed 控制 (默认 42)
- 防泄露: ODS 仅用 train 计算; val 不参与采样/分组/训练
"""
import os
import shutil
import yaml
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

BASE = r"E:\dataset_create\coco"
DS = os.path.join(BASE, "coco_person_yolo")   # person 单类 nc=1 (已预分层采样 2000 张 train + val 2693)

CLS_NAME = "person"
EPSILON = 1e-6
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
MIXED_MAX_EPOCHS = 300
PATIENCE = 20
SAMPLE_SEED = 42   # 采样固定 seed, 保证各档数据可比


def _dirs(args):
    split_tag = f"n{args.sample}"          # 分组仅由 ODS 决定, 与 seed 无关 -> 各 seed 复用同一分组
    seed_tag = f"n{args.sample}_s{args.seed}"
    return {
        'split': os.path.join(BASE, f"coco_person_split_{split_tag}"),
        'project': os.path.join(BASE, f"runs_coco_person_{seed_tag}"),
        'res': os.path.join(BASE, f"coco_person_{seed_tag}_results.txt"),
    }


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


# ==================== 采样 + 分组 ====================
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


def sample_uniform(ordered, n):
    """按 ODS 升序列表均匀间隔取 n 张, 保持难易分布覆盖 (确定性, 无随机)"""
    total = len(ordered)
    if n >= total:
        return ordered
    indices = sorted(set(int(i) for i in np.linspace(0, total - 1, n).astype(int)))
    if len(indices) < n:
        used = set(indices)
        for i in range(total):
            if len(indices) >= n:
                break
            if i not in used:
                indices.append(i)
                used.add(i)
        indices.sort()
    return [ordered[i] for i in indices]


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


def make_split(args):
    SPLIT = _dirs(args)['split']
    val_img_dir = Path(DS) / 'images' / 'val'
    if os.path.exists(SPLIT):
        shutil.rmtree(SPLIT)
    os.makedirs(SPLIT, exist_ok=True)

    ordered = load_image_ods(DS)
    full = len(ordered)
    ordered = sample_uniform(ordered, args.sample)
    total = len(ordered)
    split1 = total // 3
    split2 = 2 * total // 3
    hard = ordered[:split1]
    medium = ordered[split1:split2]
    easy = ordered[split2:]
    print(f"从 {full} 张分层采样到 {total} 张: Hard {len(hard)} / Medium {len(medium)} / Easy {len(easy)}", flush=True)

    subsets = {"Hard": hard, "Medium": medium, "Easy": easy}
    for name, items in subsets.items():
        subset_dir = os.path.join(SPLIT, name)
        for stem, ods in items:
            copy_to_subset(DS, stem, subset_dir)
        write_yaml(subset_dir, os.path.join(subset_dir, 'images', 'train'),
                   str(val_img_dir), 1, [CLS_NAME])
        print(f"  {name}: {len(items)} 张", flush=True)

    em_dir = os.path.join(SPLIT, 'EasyMedium')
    for stem, ods in easy + medium:
        copy_to_subset(DS, stem, em_dir)
    write_yaml(em_dir, os.path.join(em_dir, 'images', 'train'), str(val_img_dir), 1, [CLS_NAME])
    print(f"  EasyMedium: {len(easy) + len(medium)} 张", flush=True)

    all_dir = os.path.join(SPLIT, 'All')
    for stem, ods in easy + medium + hard:
        copy_to_subset(DS, stem, all_dir)
    write_yaml(all_dir, os.path.join(all_dir, 'images', 'train'), str(val_img_dir), 1, [CLS_NAME])
    print(f"  All: {len(easy) + len(medium) + len(hard)} 张", flush=True)
    print(f"分组完成 -> {SPLIT}", flush=True)


# ==================== 训练 ====================
def _common(data_yaml, project, name, args, patience=None):
    d = dict(
        data=data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, project=project, name=name, exist_ok=True,
        pretrained=True, optimizer='auto', verbose=False, plots=False, val=True,
        cache=('disk' if not args.no_cache else False), amp=True, lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
        box=7.5, cls=0.5, dfl=1.5, seed=args.seed,
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


def train_mixed(args, SPLIT, PROJECT):
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
    r = model.val(data=os.path.join(DS, 'data.yaml'), batch=args.batch, imgsz=args.imgsz,
                  device=args.device, workers=args.workers, verbose=False)
    m = r.box
    return {'label': label, 'mAP50': round(float(m.map50), 4),
            'mAP50-95': round(float(m.map), 4),
            'Precision': round(float(np.mean(m.p)), 4),
            'Recall': round(float(np.mean(m.r)), 4)}


def run_curriculum(budget, args, SPLIT, PROJECT):
    easy_yaml = os.path.join(SPLIT, 'Easy', 'data.yaml')
    em_yaml = os.path.join(SPLIT, 'EasyMedium', 'data.yaml')
    all_yaml = os.path.join(SPLIT, 'All', 'data.yaml')
    tag = f"b{budget}"
    print(f"\n  [课程] 预算 {budget}ep/阶段", flush=True)
    b1 = _train_one('yolov8n.pt', easy_yaml, budget, f"{tag}_stage1_easy", PROJECT, args)
    b2 = _train_one(b1, em_yaml, budget, f"{tag}_stage2_em", PROJECT, args)
    b3 = _train_one(b2, all_yaml, budget, f"{tag}_stage3_all", PROJECT, args)
    return evaluate(b3, f"{budget}ep Stage3", args)


def summarize(args, N, budgets, mixed, s3_list):
    RES_FILE = _dirs(args)['res']
    lines = ["=" * 70,
             f"COCO person 单类实验 (N={args.sample}, seed={args.seed})  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
                        choices=['ods', 'split', 'train', 'all'])
    parser.add_argument('--sample', type=int, default=2000, help='采样训练图数')
    parser.add_argument('--seed', type=int, default=42, help='训练 seed')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--no-cache', action='store_true', help='禁用磁盘缓存(cache=disk 会生成 .npy 爆盘)')
    args = parser.parse_args()

    SPLIT = _dirs(args)['split']
    PROJECT = _dirs(args)['project']

    if args.stage in ('ods', 'all'):
        print("== ODS 计算 (仅 train, 防泄露) ==", flush=True)
        compute_split_ods(DS, 'train')
    if args.stage in ('split', 'all'):
        print(f"== 采样 {args.sample} 张 + 分组 ==", flush=True)
        make_split(args)
    if args.stage in ('train', 'all'):
        print("== 训练 ==", flush=True)
        N, mixed_best = train_mixed(args, SPLIT, PROJECT)
        mixed = evaluate(mixed_best, 'Mixed', args)
        print(f"  Mixed -> {mixed}", flush=True)
        budgets = [N // 3, N // 3 + 10, N // 3 + 20]
        print(f"  档位预算: {budgets}", flush=True)
        s3_list = []
        for b in budgets:
            s3 = run_curriculum(b, args, SPLIT, PROJECT)
            s3_list.append(s3)
            print(f"  {b}ep Stage3 -> {s3}", flush=True)
        summarize(args, N, budgets, mixed, s3_list)


if __name__ == '__main__':
    main()
