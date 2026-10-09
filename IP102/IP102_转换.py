# -*- coding: utf-8 -*-
"""IP102 单类(Cicadellidae, id=101) VOC -> YOLO 转换
- 从 cicadellidae_ids.json 取 train/test id 列表
- 逐图解析 XML, 只保留 name==101 的框, 转 YOLO 格式 (nc=1)
- 不筛 size (IP102 害虫近照, 目标像素面积大, 筛 large 会删 99%)
"""
import os, re, json, shutil, yaml
from pathlib import Path

VOC = r"E:\dataset_create\IP102_det"
OUT = r"E:\dataset_create\IP102_Cicadellidae\IP102_yolo"
CLS_ID = 101

RE_WIDTH = re.compile(r'<width>\s*([\d.]+)\s*</width>', re.I)
RE_HEIGHT = re.compile(r'<height>\s*([\d.]+)\s*</height>', re.I)
RE_OBJECT = re.compile(r'<object>(.*?)</object>', re.I | re.S)
RE_NAME = re.compile(r'<name>(.*?)</name>', re.I | re.S)
RE_XMIN = re.compile(r'<xmin>\s*([\d.]+)\s*</xmin>', re.I)
RE_YMIN = re.compile(r'<ymin>\s*([\d.]+)\s*</ymin>', re.I)
RE_XMAX = re.compile(r'<xmax>\s*([\d.]+)\s*</xmax>', re.I)
RE_YMAX = re.compile(r'<ymax>\s*([\d.]+)\s*</ymax>', re.I)

ids = json.load(open(os.path.join(VOC, 'cicadellidae_ids.json')))
ann_dir = os.path.join(VOC, 'Annotations')
img_dir = os.path.join(VOC, 'JPEGImages')

def convert(ids_list, split):
    dst_img = os.path.join(OUT, 'images', split)
    dst_lbl = os.path.join(OUT, 'labels', split)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    n = 0
    for stem in ids_list:
        xml = os.path.join(ann_dir, stem + '.xml')
        img = os.path.join(img_dir, stem + '.jpg')
        if not os.path.exists(xml) or not os.path.exists(img):
            continue
        text = open(xml, encoding='utf-8', errors='ignore').read()
        mw = RE_WIDTH.search(text); mh = RE_HEIGHT.search(text)
        W = int(float(mw.group(1))) if mw else 0
        H = int(float(mh.group(1))) if mh else 0
        if W <= 0 or H <= 0:
            continue
        lines = []
        for mo in RE_OBJECT.finditer(text):
            block = mo.group(1)
            mn = RE_NAME.search(block)
            if not mn or int(mn.group(1).strip()) != CLS_ID:
                continue
            mx0 = RE_XMIN.search(block); my0 = RE_YMIN.search(block)
            mx1 = RE_XMAX.search(block); my1 = RE_YMAX.search(block)
            if not (mx0 and my0 and mx1 and my1):
                continue
            x0, y0 = float(mx0.group(1)), float(my0.group(1))
            x1, y1 = float(mx1.group(1)), float(my1.group(1))
            w = max(1.0, x1 - x0); h = max(1.0, y1 - y0)
            cx = (x0 + w / 2.0) / W; cy = (y0 + h / 2.0) / H
            bw = w / W; bh = h / H
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0); bh = min(max(bh, 0.0), 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue
        shutil.copy2(img, os.path.join(dst_img, stem + '.jpg'))
        with open(os.path.join(dst_lbl, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        n += 1
    print(f"[{split}] 转换 {n} 张")
    return n

n_train = convert(ids['train'], 'train')
n_val = convert(ids['test'], 'val')

names = ['Cicadellidae']
yaml_data = {
    'path': OUT.replace('\\', '/'),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/val',
    'nc': 1,
    'names': names,
}
with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
    yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

print(f"完成: train={n_train} val={n_val}, data.yaml 已生成 -> {OUT}")
