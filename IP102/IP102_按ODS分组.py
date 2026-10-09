# -*- coding: utf-8 -*-
"""IP102 Cicadellidae 按图级 ODS 三等分 Easy/Medium/Hard (与 Pest24 同款)
- 升序: 前 1/3 = Hard, 中 = Medium, 后 = Easy
- 生成 Easy/Medium/Hard/EasyMedium/All 子集
- val 固定指向全量 IP102_yolo 的 val
- 采样 SAMPLED_TRAIN 张 (默认 1500, 对齐 Pest24/VisDrone)
"""
import os
import shutil
import yaml
import numpy as np
from pathlib import Path

BASE = r"E:\dataset_create\IP102_Cicadellidae"
SRC = os.path.join(BASE, "IP102_yolo")
OUT = os.path.join(BASE, "IP102_split")
ods_dir_name = 'ods_values'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def load_names(src_root):
    with open(os.path.join(src_root, "data.yaml"), encoding='utf-8') as f:
        d = yaml.safe_load(f)
    return d['nc'], list(d['names'])


def load_image_ods(src_root):
    ods_dir = Path(src_root) / ods_dir_name / "train"
    items = []
    for ods_file in ods_dir.glob("*.ods"):
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
        p = Path(src_root) / "images" / "train" / f"{stem}{ext}"
        if p.exists():
            img_src = p
            break
    if img_src is None:
        return
    lbl_src = Path(src_root) / "labels" / "train" / f"{stem}.txt"
    ods_src = Path(src_root) / ods_dir_name / "train" / f"{stem}.ods"

    dst_img = Path(subset_dir) / "images" / "train"
    dst_lbl = Path(subset_dir) / "labels" / "train"
    dst_ods = Path(subset_dir) / ods_dir_name / "train"
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
    with open(Path(subset_dir) / "data.yaml", 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)


def sample_uniform(ordered, n):
    total = len(ordered)
    if n >= total:
        return ordered
    indices = np.linspace(0, total - 1, n).astype(int)
    indices = sorted(set(int(i) for i in indices))
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1500)
    args = parser.parse_args()
    SAMPLED_TRAIN = int(os.environ.get("IP102_TRAIN_N", args.n))

    nc, names = load_names(SRC)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    val_dir = os.path.join(SRC, "images", "val")

    ordered = load_image_ods(SRC)
    total_full = len(ordered)
    if total_full == 0:
        print("错误: 无 ODS, 先跑 IP102_3_计算ODS.py")
        return

    if 0 < SAMPLED_TRAIN < total_full:
        ordered = sample_uniform(ordered, SAMPLED_TRAIN)
        print(f"从 {total_full} 张分层采样到 {len(ordered)} 张训练图")
    else:
        print(f"使用全量 {total_full} 张训练图")

    total = len(ordered)
    split1 = total // 3
    split2 = 2 * total // 3
    hard = ordered[:split1]
    medium = ordered[split1:split2]
    easy = ordered[split2:]

    print(f"总 train 图片: {total}")
    print(f"  Hard(ODS最小): {len(hard)}  张, ODS ∈ [{hard[0][1]:.4f},{hard[-1][1]:.4f}]")
    print(f"  Medium:         {len(medium)}  张, ODS ∈ [{medium[0][1]:.4f},{medium[-1][1]:.4f}]")
    print(f"  Easy(ODS最大):  {len(easy)}  张, ODS ∈ [{easy[0][1]:.4f},{easy[-1][1]:.4f}]")

    subsets = {"Hard": hard, "Medium": medium, "Easy": easy}
    for name, items in subsets.items():
        subset_dir = os.path.join(OUT, name)
        for stem, ods in items:
            copy_to_subset(SRC, stem, subset_dir)
        write_yaml(subset_dir, os.path.join(subset_dir, "images", "train"), val_dir, nc, names)
        print(f"  {name} 子集: {len(items)} 张")

    em_dir = os.path.join(OUT, "EasyMedium")
    for stem, ods in easy + medium:
        copy_to_subset(SRC, stem, em_dir)
    write_yaml(em_dir, os.path.join(em_dir, "images", "train"), val_dir, nc, names)
    print(f"  EasyMedium 合并集: {len(easy) + len(medium)} 张")

    all_dir = os.path.join(OUT, "All")
    for stem, ods in easy + medium + hard:
        copy_to_subset(SRC, stem, all_dir)
    write_yaml(all_dir, os.path.join(all_dir, "images", "train"), val_dir, nc, names)
    print(f"  All 全量集: {len(easy) + len(medium) + len(hard)} 张")
    print(f"分组完成 -> {OUT}")


if __name__ == "__main__":
    main()
