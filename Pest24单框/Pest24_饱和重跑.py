# -*- coding: utf-8 -*-
"""
Pest24 饱和重跑: Mixed 训练到饱和(早停) + 课程学习三档(1/3N, 1/3N+10, 1/3N+20)
一个脚本串行跑两个实验(多框 + 单框), 全程断点续传, 供整夜 nohup 运行。

- N = Mixed 实际训练 epoch 数(早停停止时的 epoch)
- Mixed: epochs 上限 300, patience=20(连续 20 epoch 无提升即停)
- 课程学习: 3 阶段(Easy -> Easy+Medium -> All), 每阶段 = 预算
- 其余超参/ODS/分组/评估口径与之前完全一致
"""
import os
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO

BASE = r"E:\dataset_create\Pest24"

EXPERIMENTS = [
    {
        'key': 'multibox',
        'label': 'Pest24 多框版',
        'split': 'Pest24_split',
        'yolo': 'Pest24_yolo',
        'project': 'runs_pest24_saturate',
        'res': 'pest24_saturate_results.txt',
    },
    {
        'key': 'single',
        'label': 'Pest24 单框版',
        'split': 'Pest24_single_split',
        'yolo': 'Pest24_single_yolo',
        'project': 'runs_pest24_single_saturate',
        'res': 'pest24_single_saturate_results.txt',
    },
]

MIXED_MAX_EPOCHS = 300   # Mixed 训练上限(实际由 patience 早停决定)
PATIENCE = 20            # 连续 20 epoch 无提升即视为饱和


def _common(data_yaml, project, name, args, patience=None):
    d = dict(
        data=data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, project=project, name=name, exist_ok=True,
        pretrained=True, optimizer='auto', verbose=False, plots=False, val=True,
        cache=False, amp=True, lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
        box=7.5, cls=0.5, dfl=1.5,
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
    """课程学习阶段训练: 幂等 done 检查 + last.pt 断点续传"""
    last_pt = os.path.join(project, name, 'weights', 'last.pt')
    best_pt = os.path.join(project, name, 'weights', 'best.pt')
    csv_path = os.path.join(project, name, 'results.csv')
    if _read_last_epoch(csv_path) >= epochs:
        print(f"  [skip] {name} 已完成 {epochs}ep", flush=True)
        return best_pt
    common = _common(data_yaml, project, name, args)
    if os.path.exists(last_pt):
        print(f"  [resume] {name} 从断点续训至 {epochs}ep", flush=True)
        model = YOLO(last_pt)
        model.train(epochs=epochs, resume=True, **common)
    else:
        print(f"  [start] {name} 从头训 {epochs}ep", flush=True)
        model = YOLO(model_path)
        model.train(epochs=epochs, **common)
    return best_pt


def train_mixed(cfg, args):
    """训练 Mixed 到饱和(早停), 返回 (N, best_pt)"""
    project = os.path.join(BASE, cfg['project'])
    os.makedirs(project, exist_ok=True)
    flag = os.path.join(project, 'mixed_saturated.txt')
    best_pt = os.path.join(project, 'mixed', 'weights', 'best.pt')
    if os.path.exists(flag):
        N = int(open(flag).read().strip())
        print(f"  Mixed 已饱和训练完成 (N={N}), 跳过训练", flush=True)
        return N, best_pt

    all_yaml = os.path.join(BASE, cfg['split'], 'All', 'data.yaml')
    last_pt = os.path.join(project, 'mixed', 'weights', 'last.pt')
    common = _common(all_yaml, project, 'mixed', args, patience=PATIENCE)
    if os.path.exists(last_pt):
        print(f"  [resume] Mixed 从断点续训至饱和", flush=True)
        model = YOLO(last_pt)
        model.train(epochs=MIXED_MAX_EPOCHS, resume=True, **common)
    else:
        print(f"  [start] Mixed 训练至饱和 (上限 {MIXED_MAX_EPOCHS}ep, patience={PATIENCE})", flush=True)
        model = YOLO('yolov8n.pt')
        model.train(epochs=MIXED_MAX_EPOCHS, **common)

    csv_path = os.path.join(project, 'mixed', 'results.csv')
    N = _read_last_epoch(csv_path)
    with open(flag, 'w') as f:
        f.write(str(N))
    print(f"  Mixed 饱和训练完成, 实际 N={N}", flush=True)
    return N, best_pt


def evaluate(model_path, label, args, val_yaml):
    model = YOLO(model_path)
    r = model.val(data=val_yaml, batch=args.batch, imgsz=args.imgsz,
                  device=args.device, workers=args.workers, verbose=False)
    m = r.box
    return {'label': label,
            'mAP50': round(float(m.map50), 4),
            'mAP50-95': round(float(m.map), 4),
            'Precision': round(float(np.mean(m.p)), 4),
            'Recall': round(float(np.mean(m.r)), 4)}


def run_curriculum(cfg, budget, args, val_yaml):
    project = os.path.join(BASE, cfg['project'])
    easy_yaml = os.path.join(BASE, cfg['split'], 'Easy', 'data.yaml')
    em_yaml = os.path.join(BASE, cfg['split'], 'EasyMedium', 'data.yaml')
    all_yaml = os.path.join(BASE, cfg['split'], 'All', 'data.yaml')
    tag = f"b{budget}"
    print(f"\n  [课程] 预算 {budget}ep/阶段 (3 阶段)", flush=True)
    best_s1 = _train_one('yolov8n.pt', easy_yaml, budget, f"{tag}_stage1_easy", project, args)
    best_s2 = _train_one(best_s1, em_yaml, budget, f"{tag}_stage2_em", project, args)
    best_s3 = _train_one(best_s2, all_yaml, budget, f"{tag}_stage3_all", project, args)
    return evaluate(best_s3, f"{budget}ep Stage3", args, val_yaml)


def summarize(cfg, N, budgets, mixed, s3_list, args):
    lines = []
    lines.append("=" * 70)
    lines.append(f"{cfg['label']} 饱和重跑结果  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(f"Mixed 饱和训练实际 epoch N = {N} (早停 patience={PATIENCE})")
    lines.append(f"课程学习三档预算: {budgets[0]}ep / {budgets[1]}ep / {budgets[2]}ep")
    lines.append(f"{'配置':<20}{'mAP50':>10}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}")
    lines.append("-" * 70)
    lines.append(f"{mixed['label']:<20}{mixed['mAP50']:>10}{mixed['mAP50-95']:>12}"
                 f"{mixed['Precision']:>12}{mixed['Recall']:>10}")
    for s3 in s3_list:
        lines.append(f"{s3['label']:<20}{s3['mAP50']:>10}{s3['mAP50-95']:>12}"
                     f"{s3['Precision']:>12}{s3['Recall']:>10}")
    lines.append("")
    lines.append("增益 (Stage3 mAP50 - Mixed mAP50):")
    for b, s3 in zip(budgets, s3_list):
        g = s3['mAP50'] - mixed['mAP50']
        lines.append(f"  {b}ep: {g:+.4f}")
    out = os.path.join(BASE, cfg['res'])
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)


def run_experiment(cfg, args):
    print("\n" + "#" * 70, flush=True)
    print(f"# {cfg['label']}", flush=True)
    print("#" * 70, flush=True)
    val_yaml = os.path.join(BASE, cfg['yolo'], 'data.yaml')

    N, mixed_best = train_mixed(cfg, args)
    mixed = evaluate(mixed_best, 'Mixed', args, val_yaml)
    print(f"  Mixed -> {mixed}", flush=True)

    budgets = [N // 3, N // 3 + 10, N // 3 + 20, N // 3 + 30]
    print(f"  三档预算: {budgets}", flush=True)
    s3_list = []
    for b in budgets:
        s3 = run_curriculum(cfg, b, args, val_yaml)
        s3_list.append(s3)
        print(f"  {b}ep Stage3 -> {s3}", flush=True)

    summarize(cfg, N, budgets, mixed, s3_list, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default='', help='只跑 multibox 或 single')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    print(f"batch={args.batch}, workers={args.workers}, device={args.device}, "
          f"patience={PATIENCE}, mixed_max={MIXED_MAX_EPOCHS}", flush=True)

    for cfg in EXPERIMENTS:
        if args.only and cfg['key'] != args.only:
            continue
        run_experiment(cfg, args)

    print("\n全部实验完成", flush=True)


if __name__ == '__main__':
    main()
