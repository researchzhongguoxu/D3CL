"""
VisDrone2019-DET 标注转 YOLO 格式
=====================================
VisDrone 原始标注每行: bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion

规则:
  - 忽略 category 0 (ignored regions) 和 category 11 (others)i --
  - 忽略 score == 0 的标注 (VisDrone 官方评估只统计 score>0)
  - 保留 category 1-10, 映射到 YOLO class 0-9
  - 坐标归一化到 [0,1], 输出 YOLO 格式: class cx cy w h

类别 (category -> YOLO id):
  1 pedestrian -> 0
  2 people -> 1
  3 bicycle -> 2
  4 car -> 3
  5 van -> 4
  6 truck -> 5
  7 tricycle -> 6
  8 awning-tricycle -> 7
  9 bus -> 8
  10 motor -> 9
"""
import os
import cv2
import shutil
from pathlib import Path

# VisDrone 有效类别: category id (1-10) -> YOLO class id (0-9)
CATEGORY_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

VISDRONE_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def convert_split(src_root, split, dst_root):
    """
    转换一个 split (train/val) 的 VisDrone 标注到 YOLO 格式。
    src_root: VisDrone2019-DET-{split} 解压根目录 (含 images/ 和 annotations/)
    dst_root: 输出 YOLO 数据集根目录 (会创建 images/{split} 和 labels/{split})
    """
    src_img_dir = Path(src_root) / "images"
    src_ann_dir = Path(src_root) / "annotations"

    if not src_img_dir.exists() or not src_ann_dir.exists():
        raise FileNotFoundError(f"找不到源目录: {src_root}")

    dst_img_dir = Path(dst_root) / "images" / split
    dst_lbl_dir = Path(dst_root) / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in src_img_dir.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS])

    n_skipped_img = 0
    n_kept_box = 0
    n_dropped_box = 0

    for img_path in img_files:
        stem = img_path.stem
        ann_path = src_ann_dir / f"{stem}.txt"

        if not ann_path.exists():
            n_skipped_img += 1
            continue

        # 读取图像尺寸 (VisDrone 图片尺寸不统一)
        img = cv2.imread(str(img_path))
        if img is None:
            n_skipped_img += 1
            continue
        h, w = img.shape[:2]

        # 转换标注
        yolo_lines = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 6:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                score = float(parts[4])
                category = int(parts[5])

                # 过滤: score=0 或无效类别
                if score == 0 or category not in CATEGORY_MAP:
                    n_dropped_box += 1
                    continue

                cls = CATEGORY_MAP[category]

                # VisDrone (左上角+宽高) -> YOLO (归一化中心+宽高)
                cx = (bbox_left + bbox_w / 2.0) / w
                cy = (bbox_top + bbox_h / 2.0) / h
                bw = bbox_w / w
                bh = bbox_h / h

                # 裁剪到 [0,1] 防止越界
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)

                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_kept_box += 1

        # 复制图片 + 写标签
        shutil.copy2(str(img_path), str(dst_img_dir / img_path.name))
        lbl_path = dst_lbl_dir / f"{stem}.txt"
        with open(lbl_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

    print(f"[{split}] 图片: {len(img_files)} 张 (跳过 {n_skipped_img})")
    print(f"[{split}] 保留目标框: {n_kept_box}, 丢弃(score=0/cat0/cat11): {n_dropped_box}")


if __name__ == "__main__":
    BASE = r"E:\dataset_create\visdrone"
    OUT = os.path.join(BASE, "VisDrone_yolo")

    convert_split(
        src_root=os.path.join(BASE, "train_extract", "VisDrone2019-DET-train"),
        split="train",
        dst_root=OUT,
    )
    convert_split(
        src_root=os.path.join(BASE, "val_extract", "VisDrone2019-DET-val"),
        split="val",
        dst_root=OUT,
    )

    # 生成 data.yaml (先占位, 后续分组再补全)
    import yaml
    yaml_data = {
        'path': OUT.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'nc': 10,
        'names': VISDRONE_NAMES,
    }
    with open(os.path.join(OUT, "data.yaml"), 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n转换完成, 数据集输出至: {OUT}")
