# -*- coding: utf-8 -*-
"""生成 val150/test150 子集, 并更新所有 data.yaml 的 val/test 指向。
- 从原全量 val 随机采样 150 张作为 val150, 另 150 张作为 test150 (固定 seed 可复现)
- 更新: {yolo}/data.yaml + {split}/{Easy,Medium,Hard,EasyMedium,All}/data.yaml 的 val/test
"""
import os
import random
import shutil
import yaml
from pathlib import Path

BASE = r"E:\dataset_create\Pest24"
random.seed(42)
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
N_VAL = 150
N_TEST = 150

CONFIGS = [
    {'yolo': 'Pest24_yolo', 'split': 'Pest24_split'},
    {'yolo': 'Pest24_single_yolo', 'split': 'Pest24_single_split'},
]


def find_img(img_dir, stem):
    for ext in IMAGE_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def gen_subsets(yolo):
    img_dir = Path(BASE) / yolo / 'images' / 'val'
    lbl_dir = Path(BASE) / yolo / 'labels' / 'val'
    stems = sorted([f.stem for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    random.shuffle(stems)
    val_stems = stems[:N_VAL]
    test_stems = stems[N_VAL:N_VAL + N_TEST]

    for name, subset in [('val150', val_stems), ('test150', test_stems)]:
        dst_img = Path(BASE) / yolo / name / 'images'
        dst_lbl = Path(BASE) / yolo / name / 'labels'
        for d in (dst_img, dst_lbl):
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                f.unlink()
        for s in subset:
            src_img = find_img(img_dir, s)
            if src_img is None:
                continue
            shutil.copy2(src_img, dst_img / src_img.name)
            lbl = lbl_dir / f"{s}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / f"{s}.txt")
    return len(val_stems), len(test_stems)


def update_yaml(yaml_path, val_img_dir, test_img_dir):
    with open(yaml_path, encoding='utf-8') as f:
        d = yaml.safe_load(f)
    d['val'] = str(val_img_dir).replace('\\', '/')
    d['test'] = str(test_img_dir).replace('\\', '/')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(d, f, default_flow_style=False, allow_unicode=True)


def main():
    for cfg in CONFIGS:
        yolo, split = cfg['yolo'], cfg['split']
        n_val, n_test = gen_subsets(yolo)
        val_img = Path(BASE) / yolo / 'val150' / 'images'
        test_img = Path(BASE) / yolo / 'test150' / 'images'
        print(f"[{yolo}] val150={n_val}, test150={n_test}")

        update_yaml(Path(BASE) / yolo / 'data.yaml', val_img, test_img)
        for sub in ['Easy', 'Medium', 'Hard', 'EasyMedium', 'All']:
            p = Path(BASE) / split / sub / 'data.yaml'
            if p.exists():
                update_yaml(p, val_img, test_img)
        print(f"[{yolo}] 已更新 {yolo}/data.yaml 及 {split}/*/data.yaml 的 val/test")
    print("完成")


if __name__ == '__main__':
    main()
