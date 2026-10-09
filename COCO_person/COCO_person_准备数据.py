# -*- coding: utf-8 -*-
"""
COCO person 单类准备: 分层采样 train 2000 张 + val 全量 2693 张 -> 下载图片 -> YOLO 格式
- person (category_id=1) 单类, 忽略 iscrowd
- train 按"每图 person 框平均 S=sqrt(area)"排序后均匀间隔采样 2000 张 (覆盖小->大完整谱系)
- val 全用 (2693 张)
- 多线程下载, 失败重试
"""
import json
import os
import sys
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = r'E:\dataset_create\coco'
DS = os.path.join(BASE, 'coco_person_yolo')
TRAIN_N = 2000
SEED = 42
IMG_ROOT = 'http://images.cocodataset.org'
WORKERS = 16


def load_person(ann_path):
    d = json.load(open(ann_path, encoding='utf-8'))
    img_info = {i['id']: i for i in d['images']}
    img_person = defaultdict(list)
    for a in d['annotations']:
        if a['category_id'] != 1:
            continue
        if a.get('iscrowd', 0):
            continue
        img_person[a['image_id']].append(a)
    return img_info, img_person


def sample_uniform(items, n):
    total = len(items)
    if n >= total:
        return items
    idx = sorted(set(int(i) for i in np.linspace(0, total - 1, n)))
    return [items[i] for i in idx]


def main():
    print("== 加载 train annotations ==", flush=True)
    img_info_tr, img_person_tr = load_person(os.path.join(BASE, 'annotations', 'instances_train2017.json'))
    print(f"  train 含 person 图 {len(img_person_tr)} 张", flush=True)

    items = []
    for img_id, anns in img_person_tr.items():
        S = float(np.mean([np.sqrt(a['bbox'][2] * a['bbox'][3]) for a in anns]))
        items.append((img_id, S, len(anns)))
    items.sort(key=lambda x: x[1])
    train_items = sample_uniform(items, TRAIN_N)
    print(f"  分层采样 train {len(train_items)} 张 (S 范围 {train_items[0][1]:.1f} ~ {train_items[-1][1]:.1f}px)", flush=True)

    print("== 加载 val annotations ==", flush=True)
    img_info_val, img_person_val = load_person(os.path.join(BASE, 'annotations', 'instances_val2017.json'))
    val_items = [(iid, 0.0, len(anns)) for iid, anns in img_person_val.items()]
    print(f"  val 含 person 图 {len(val_items)} 张 (全用)", flush=True)

    # 下载 + 转 YOLO
    tasks = []
    for split, items_list, img_info, img_person, sub in [
        ('train', train_items, img_info_tr, img_person_tr, 'train2017'),
        ('val', val_items, img_info_val, img_person_val, 'val2017'),
    ]:
        dst_img = Path(DS) / 'images' / split
        dst_lbl = Path(DS) / 'labels' / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for (img_id, S, nbox) in items_list:
            info = img_info[img_id]
            fn = info['file_name']
            w, h = info['width'], info['height']
            anns = img_person[img_id]
            tasks.append((sub, fn, w, h, anns, str(dst_img / fn), str(dst_lbl / (Path(fn).stem + '.txt'))))

    print(f"== 下载 {len(tasks)} 张图 (16 线程) ==", flush=True)

    def work(t):
        sub, fn, w, h, anns, dst_img, dst_lbl = t
        url = f"{IMG_ROOT}/{sub}/{fn}"
        ok = False
        for _ in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    with open(dst_img, 'wb') as f:
                        f.write(r.content)
                    ok = True
                    break
            except Exception:
                pass
        if not ok:
            return (fn, False)
        # 写 label
        lines = []
        for a in anns:
            x, y, bw, bh = a['bbox']
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(dst_lbl, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        return (fn, True)

    fail = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            fn, ok = fut.result()
            done += 1
            if not ok:
                fail.append(fn)
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} 完成", flush=True)
    print(f"下载完成, 失败 {len(fail)} 张", flush=True)
    if fail:
        print("失败列表:", fail[:20], flush=True)

    # 写 data.yaml
    yaml_data = {
        'path': DS.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'nc': 1,
        'names': ['person'],
    }
    import yaml as y
    with open(os.path.join(DS, 'data.yaml'), 'w', encoding='utf-8') as f:
        y.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    print(f"完成 -> {DS}", flush=True)


if __name__ == '__main__':
    main()
