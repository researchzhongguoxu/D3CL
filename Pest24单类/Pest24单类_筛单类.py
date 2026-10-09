# -*- coding: utf-8 -*-
"""从 Pest24_single_yolo 筛选单类别(id=5 东方蝼蛄)的单框图 -> Pest24_single_cls5_yolo
- 只保留 label 恰好 1 行且 class id=5 的图
- 类别 id 重映射为 0 (单类, nc=1)
"""
import os
import shutil
import yaml
from pathlib import Path

SRC = r"E:\dataset_create\Pest24\Pest24_single_yolo"
OUT = r"E:\dataset_create\Pest24\Pest24_single_cls5_yolo"
CLS_ID = 5
CLS_NAME = "Gryllotalpa orientalis"
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def find_img(img_dir, stem):
    for ext in IMAGE_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    total = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        lbl_src = Path(SRC) / 'labels' / split
        img_src = Path(SRC) / 'images' / split
        dst_img = Path(OUT) / 'images' / split
        dst_lbl = Path(OUT) / 'labels' / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for lbl in lbl_src.glob('*.txt'):
            lines = [l for l in lbl.read_text(encoding='utf-8').splitlines() if l.strip()]
            if len(lines) != 1:
                continue
            parts = lines[0].split()
            if len(parts) < 5 or int(float(parts[0])) != CLS_ID:
                continue
            img = find_img(img_src, lbl.stem)
            if img is None:
                continue
            shutil.copy2(str(img), str(dst_img / img.name))
            # 重映射 class id -> 0
            new_line = "0 " + " ".join(parts[1:5]) + "\n"
            (dst_lbl / f"{lbl.stem}.txt").write_text(new_line, encoding='utf-8')
            total[split] += 1
        print(f"[{split}] 单类(id={CLS_ID})单框图 {total[split]} 张")

    yaml_data = {
        'path': OUT.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'nc': 1,
        'names': [CLS_NAME],
    }
    with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    print(f"完成 -> {OUT} (nc=1, train={total['train']}, val={total['val']})")


if __name__ == '__main__':
    main()
