"""
按图级 ODS 把 Pest24 train 分成 Easy/Medium/Hard (quantile 三等分)
=====================================
- 读 ods_values/train 里的图级 ODS
- 按 ODS 升序排序: 前 1/3 = Hard(最难), 中 1/3 = Medium, 后 1/3 = Easy(最易)
- 生成 Easy/Medium/Hard 三个子集 (物理复制 images + labels + ods_values)
- 额外生成 EasyMedium 合并集 (供 Stage2 课程学习)
- val 保持固定 (统一指向全量 Pest24_yolo 的 val), 保证公平评估
- 训练规模: 默认从全量 train 分层采样到 PEST24_TRAIN_N 张 (默认 1500, 对齐 VisDrone)
"""
import os
import shutil
import yaml
import numpy as np
from pathlib import Path

BASE = r"E:\dataset_create\Pest24"
SRC = os.path.join(BASE, "Pest24_yolo")
OUT = os.path.join(BASE, "Pest24_split")
ods_dir_name = 'ods_values'

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def load_names(src_root):
    """从 data.yaml 读 nc + names (由 1_转换并筛选Pest24.py 生成)。"""
    yaml_path = os.path.join(src_root, "data.yaml")
    with open(yaml_path, encoding='utf-8') as f:
        d = yaml.safe_load(f)
    return d['nc'], list(d['names'])


def load_image_ods(src_root):
    """返回 [(stem, ods), ...] 按 ODS 升序排序"""
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
    """分层采样: 从按 ODS 排序的列表中均匀间隔取 n 张, 保持 ODS 分布覆盖。"""
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
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--n', type=int, default=1500, help='训练采样张数 (默认 1500, 对齐 VisDrone)')
    args = parser.parse_args()
    SAMPLED_TRAIN = int(os.environ.get("PEST24_TRAIN_N", args.n))

    nc, names = load_names(SRC)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    val_dir = os.path.join(SRC, "images", "val")

    ordered = load_image_ods(SRC)
    total_full = len(ordered)
    if total_full == 0:
        print("错误: 没有可用的图级 ODS。请先运行 2_计算ODS.py")
        return

    if 0 < SAMPLED_TRAIN < total_full:
        ordered = sample_uniform(ordered, SAMPLED_TRAIN)
        print(f"从 {total_full} 张分层采样到 {len(ordered)} 张训练图 (保持 ODS 分布)")
    else:
        print(f"使用全量 {total_full} 张训练图 (不采样)")

    total = len(ordered)
    split1 = total // 3
    split2 = 2 * total // 3

    hard = ordered[:split1]
    medium = ordered[split1:split2]
    easy = ordered[split2:]

    print(f"总 train 图片: {total}")
    print(f"  Hard   (ODS 最小): {len(hard)}  张, ODS ∈ [{hard[0][1]:.4f}, {hard[-1][1]:.4f}]")
    print(f"  Medium:            {len(medium)}  张, ODS ∈ [{medium[0][1]:.4f}, {medium[-1][1]:.4f}]")
    print(f"  Easy   (ODS 最大): {len(easy)}  张, ODS ∈ [{easy[0][1]:.4f}, {easy[-1][1]:.4f}]")

    subsets = {"Hard": hard, "Medium": medium, "Easy": easy}
    for name, items in subsets.items():
        subset_dir = os.path.join(OUT, name)
        for stem, ods in items:
            copy_to_subset(SRC, stem, subset_dir)
        write_yaml(subset_dir, os.path.join(subset_dir, "images", "train"), val_dir, nc, names)
        print(f"  {name} 子集已生成: {len(items)} 张")

    em_dir = os.path.join(OUT, "EasyMedium")
    for stem, ods in easy + medium:
        copy_to_subset(SRC, stem, em_dir)
    write_yaml(em_dir, os.path.join(em_dir, "images", "train"), val_dir, nc, names)
    print(f"  EasyMedium 合并集已生成: {len(easy) + len(medium)} 张")

    all_dir = os.path.join(OUT, "All")
    for stem, ods in easy + medium + hard:
        copy_to_subset(SRC, stem, all_dir)
    write_yaml(all_dir, os.path.join(all_dir, "images", "train"), val_dir, nc, names)
    print(f"  All 全量集已生成: {len(easy) + len(medium) + len(hard)} 张 (采样后全集)")

    print(f"\n分组完成, 输出目录: {OUT}")


if __name__ == "__main__":
    main()
