"""
Pest24 标注转 YOLO 格式 + 筛选中小目标
=====================================
Pest24 是 VOC2007 格式 (Annotations/XML + JPEGImages/JPG + ImageSets/Main/*.txt)。

流程:
  1. 解析 VOC XML -> 提取 (width, height, 每个 object 的 name + bndbox)
  2. 从所有 XML 的 <name> 里收集类别名, 排序 -> 稳定 class id 0..23
  3. 按 COCO 标准把目标按面积分小/中/大:
       small  : area <  32^2 = 1024        (保留)
       medium : 1024 <= area < 96^2 = 9216 (保留)
       large  : area >= 9216               (丢弃)
     只保留 small + medium, 丢弃 large; 目标被清空的图片整张丢弃。
  4. 转 YOLO 格式 (class cx cy w h, 归一化), 物理复制图片 + 写标签
  5. 生成 data.yaml (nc=24, names 为收集到的类别名)

用法:
  python "1_转换并筛选Pest24.py"
"""
import os
import re
import shutil
import yaml
from pathlib import Path

VOC_DIR = r"E:\dataset_create\Pest24\VOCdevkit\voc2007"
OUT = r"E:\dataset_create\Pest24\Pest24_yolo"

# COCO 大小阈值 (面积 = 像素宽 * 像素高)
SMALL_MAX = 32 ** 2      # 1024
MEDIUM_MAX = 96 ** 2     # 9216

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 正则解析 (对个别格式不规范的 XML 更鲁棒)
RE_WIDTH = re.compile(r'<width>\s*([\d.]+)\s*</width>', re.I)
RE_HEIGHT = re.compile(r'<height>\s*([\d.]+)\s*</height>', re.I)
RE_OBJECT = re.compile(r'<object>(.*?)</object>', re.I | re.S)
RE_NAME = re.compile(r'<name>(.*?)</name>', re.I | re.S)
RE_XMIN = re.compile(r'<xmin>\s*([\d.]+)\s*</xmin>', re.I)
RE_YMIN = re.compile(r'<ymin>\s*([\d.]+)\s*</ymin>', re.I)
RE_XMAX = re.compile(r'<xmax>\s*([\d.]+)\s*</xmax>', re.I)
RE_YMAX = re.compile(r'<ymax>\s*([\d.]+)\s*</ymax>', re.I)


def parse_annotation(xml_path):
    """返回 (width, height, [(name, xmin, ymin, xmax, ymax), ...])"""
    text = xml_path.read_text(encoding='utf-8', errors='ignore')
    mw = RE_WIDTH.search(text)
    mh = RE_HEIGHT.search(text)
    width = int(float(mw.group(1))) if mw else 0
    height = int(float(mh.group(1))) if mh else 0

    objs = []
    for mo in RE_OBJECT.finditer(text):
        block = mo.group(1)
        mn = RE_NAME.search(block)
        if not mn:
            continue
        name = mn.group(1).strip()
        mx0 = RE_XMIN.search(block)
        my0 = RE_YMIN.search(block)
        mx1 = RE_XMAX.search(block)
        my1 = RE_YMAX.search(block)
        if not (mx0 and my0 and mx1 and my1):
            continue
        objs.append((name,
                     float(mx0.group(1)), float(my0.group(1)),
                     float(mx1.group(1)), float(my1.group(1))))
    return width, height, objs


def collect_names(ann_dir):
    """遍历所有 XML, 收集全部类别名 (去重排序)。"""
    names = set()
    for xml_path in ann_dir.glob('*.xml'):
        _, _, objs = parse_annotation(xml_path)
        for name, *_ in objs:
            names.add(name)
    return sorted(names)


def read_ids(txt_file):
    """读 ImageSets/Main 里的 id 列表 (兼容 'id' 与 'id 1' 两种写法)。"""
    ids = []
    with open(txt_file, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line.split()[0])
    return list(dict.fromkeys(ids))


def discover_splits(voc_dir):
    """
    确定 train / val 图片 id 列表。
    优先级: train.txt+val.txt -> trainval.txt+test.txt -> 全部图片 8:2 切分。
    """
    main_dir = Path(voc_dir) / 'ImageSets'
    txt_files = {f.stem.lower(): f for f in main_dir.glob('*.txt')} if main_dir.exists() else {}

    splits = {}
    if 'train' in txt_files and 'val' in txt_files:
        splits['train'] = read_ids(txt_files['train'])
        splits['val'] = read_ids(txt_files['val'])
        print("ImageSets: 使用 train.txt + val.txt")
    elif 'trainval' in txt_files and 'test' in txt_files:
        splits['train'] = read_ids(txt_files['trainval'])
        splits['val'] = read_ids(txt_files['test'])
        print("ImageSets: 使用 trainval.txt -> train, test.txt -> val")
    else:
        ann_dir = Path(voc_dir) / 'Annotations'
        all_ids = sorted(p.stem for p in ann_dir.glob('*.xml'))
        k = int(len(all_ids) * 0.8)
        splits['train'] = all_ids[:k]
        splits['val'] = all_ids[k:]
        print("ImageSets: 未找到 train/val 划分, 按 8:2 从全部图片切分")

    return splits


def convert_split(voc_dir, dst_root, split, ids, name_to_id):
    ann_dir = Path(voc_dir) / 'Annotations'
    img_dir = Path(voc_dir) / 'images'

    # 一次性建立 stem -> 图片路径 映射 (大小写不敏感, 避免 .JPG/.jpg 混用漏图)
    img_map = {}
    for p in img_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            img_map[p.stem] = p

    dst_img = Path(dst_root) / 'images' / split
    dst_lbl = Path(dst_root) / 'labels' / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_skipped_img = 0
    n_small = n_medium = n_large = 0

    for stem in ids:
        xml_path = ann_dir / f'{stem}.xml'
        if not xml_path.exists():
            n_skipped_img += 1
            continue

        img_path = img_map.get(stem)
        if img_path is None:
            n_skipped_img += 1
            continue

        width, height, objs = parse_annotation(xml_path)
        if width <= 0 or height <= 0:
            n_skipped_img += 1
            continue

        yolo_lines = []
        for name, x0, y0, x1, y1 in objs:
            w = max(1.0, x1 - x0)
            h = max(1.0, y1 - y0)
            area = w * h

            if area >= MEDIUM_MAX:
                n_large += 1
                continue
            elif area < SMALL_MAX:
                n_small += 1
            else:
                n_medium += 1

            cls = name_to_id[name]
            cx = (x0 + w / 2.0) / width
            cy = (y0 + h / 2.0) / height
            bw = w / width
            bh = h / height
            # 裁剪到 [0,1] 防止越界
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            # 过滤后没有任何中小目标, 整张丢弃
            n_skipped_img += 1
            continue

        # 统一输出为小写扩展名, 保证后续脚本 (按 IMAGE_EXTS 查找) 能命中
        out_name = f'{stem}{img_path.suffix.lower()}'
        shutil.copy2(str(img_path), str(dst_img / out_name))
        with open(dst_lbl / f'{stem}.txt', 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines) + "\n")
        n_img += 1

    print(f"[{split}] 图片: {n_img} 张 (丢弃 {n_skipped_img})  "
          f"保留框: small={n_small} medium={n_medium}  丢弃 large={n_large}")
    return n_img


def main():
    voc_dir = Path(VOC_DIR)
    ann_dir = voc_dir / 'Annotations'
    if not ann_dir.exists():
        raise FileNotFoundError(f"找不到标注目录: {ann_dir} (请先解压 Pest24.zip)")

    # 1. 收集类别名 -> 稳定映射
    names = collect_names(ann_dir)
    name_to_id = {n: i for i, n in enumerate(names)}
    print(f"检测到 {len(names)} 个类别:")
    for i, n in enumerate(names):
        print(f"  {i:>2}: {n}")

    # 2. 确定划分
    splits = discover_splits(voc_dir)
    for k, v in splits.items():
        print(f"  split '{k}': {len(v)} 张")

    # 3. 转换 + 筛选
    os.makedirs(OUT, exist_ok=True)
    for split, ids in splits.items():
        convert_split(voc_dir, OUT, split, ids, name_to_id)

    # 4. 生成 data.yaml
    yaml_data = {
        'path': OUT.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'nc': len(names),
        'names': names,
    }
    with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n转换完成, 数据集输出至: {OUT}")


if __name__ == "__main__":
    main()
