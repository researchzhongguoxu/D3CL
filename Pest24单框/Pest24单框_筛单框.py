# -*- coding: utf-8 -*-
"""从 Pest24_yolo 筛出单框(恰好1个框)图 -> Pest24_single_yolo
- 原 Pest24_yolo 已只含 small+medium 框(large 已筛掉)
- 单框图: label txt 恰好 1 行
- 复制 images + labels + 重建 data.yaml(仍 nc=24)
"""
import os, shutil, yaml
from pathlib import Path

SRC = r"E:\dataset_create\Pest24\Pest24_yolo"
OUT = r"E:\dataset_create\Pest24\Pest24_single_yolo"
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 读原 data.yaml 拿 nc/names
with open(os.path.join(SRC, 'data.yaml'), encoding='utf-8') as f:
    d = yaml.safe_load(f)
nc, names = d['nc'], list(d['names'])

for split in ['train', 'val']:
    lbl_src = Path(SRC) / 'labels' / split
    img_src = Path(SRC) / 'images' / split
    dst_img = Path(OUT) / 'images' / split
    dst_lbl = Path(OUT) / 'labels' / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n = 0
    for lbl in lbl_src.glob('*.txt'):
        lines = [l for l in lbl.read_text(encoding='utf-8').splitlines() if l.strip()]
        if len(lines) != 1:
            continue
        stem = lbl.stem
        img = None
        for ext in IMAGE_EXTS:
            p = img_src / f"{stem}{ext}"
            if p.exists():
                img = p
                break
        if img is None:
            continue
        shutil.copy2(str(img), str(dst_img / img.name))
        shutil.copy2(str(lbl), str(dst_lbl / f"{stem}.txt"))
        n += 1
    print(f"[{split}] 单框图 {n} 张")

yaml_data = {
    'path': OUT.replace('\\', '/'),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/val',
    'nc': nc,
    'names': names,
}
with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
    yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
print(f"完成 -> {OUT} (nc={nc})")
