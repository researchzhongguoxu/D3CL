"""
VisDrone 实验训练脚本: Mixed 基线 vs ODS 课程学习 (支持断点续传)
=====================================
- Detector: YOLOv8n (与主实验一致)
- 课程学习: Stage1 Easy -> Stage2 Easy+Medium -> Stage3 全量
- Mixed 基线: 全量混合训练 (总 epochs 与课程学习对齐)
- 评估: 固定 val 集上的 mAP50 / mAP50-95 / Precision / Recall

断点续传: 每个阶段若存在 weights/last.pt, 自动 resume 从断点继续; 否则从头训练。
用法:
  python visdrone_train.py                     # 跑全部 4 个阶段
  python visdrone_train.py --stage mixed       # 只跑 Mixed 阶段
  python visdrone_train.py --stage curriculum  # 只跑课程学习 3 阶段
"""
import os
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO

BASE = r"E:\dataset_create\visdrone"
SPLIT = os.path.join(BASE, "VisDrone_split")
YOLO_DS = os.path.join(BASE, "VisDrone_yolo")
PROJECT = os.path.join(BASE, "runs_visdrone")
# Mixed 基线固定 120ep, 只训练一次, 不随 --tag 变化
MIXED_BEST = os.path.join(BASE, "runs_visdrone", "mixed", "weights", "best.pt")

VAL_YAML = os.path.join(YOLO_DS, "data.yaml")
EASY_YAML = os.path.join(SPLIT, "Easy", "data.yaml")
EM_YAML = os.path.join(SPLIT, "EasyMedium", "data.yaml")
ALL_YAML = os.path.join(SPLIT, "All", "data.yaml")


def _train_one(model_path, data_yaml, epochs, name, batch, imgsz, device, workers):
    """训练一个阶段; 若该阶段已有 last.pt 则 resume。返回 best.pt 路径。"""
    last_pt = os.path.join(PROJECT, name, "weights", "last.pt")

    common = dict(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=PROJECT,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=False,
        plots=False,
        val=True,
        cache='disk',
        amp=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    if os.path.exists(last_pt):
        print(f"  [resume] 检测到 {name} 的 last.pt, 从断点继续训练...")
        model = YOLO(last_pt)
        model.train(epochs=epochs, resume=True, **common)
    else:
        print(f"  [start] {name} 从头训练...")
        model = YOLO(model_path)
        model.train(epochs=epochs, **common)

    return os.path.join(PROJECT, name, "weights", "best.pt")


def evaluate(model_path, label, batch, imgsz, device, workers):
    model = YOLO(model_path)
    r = model.val(data=VAL_YAML, batch=batch, imgsz=imgsz, device=device,
                  workers=workers, verbose=False)
    m = r.box
    return {
        'label': label,
        'mAP50': round(float(m.map50), 4),
        'mAP50-95': round(float(m.map), 4),
        'Precision': round(float(np.mean(m.p)), 4),
        'Recall': round(float(np.mean(m.r)), 4),
    }


def run_mixed(args):
    print(f"\n[1] Mixed 基线训练 (全量混合, {args.mixed_epochs} epochs)...")
    best = os.path.join(PROJECT, "mixed", "weights", "best.pt")
    csv_path = os.path.join(PROJECT, "mixed", "results.csv")
    done = False
    if os.path.exists(csv_path) and os.path.exists(best):
        try:
            with open(csv_path) as f:
                last = f.readlines()[-1].split(',')[0]
            done = int(last) >= args.mixed_epochs
        except Exception:
            done = False
    if done:
        print("  Mixed 已训练完成, 跳过训练, 直接评估...")
    else:
        best = _train_one("yolov8n.pt", ALL_YAML, args.mixed_epochs, "mixed",
                          args.batch, args.imgsz, args.device, args.workers)
    return evaluate(best, "Mixed", args.batch, args.imgsz, args.device, args.workers)


def run_curriculum(args):
    print(f"\n[2] ODS 课程学习 (3 阶段 × {args.stage_epochs} epochs)...")
    print("  Stage1: Easy")
    best_s1 = _train_one("yolov8n.pt", EASY_YAML, args.stage_epochs,
                         "curriculum_stage1_easy", args.batch, args.imgsz,
                         args.device, args.workers)
    s1 = evaluate(best_s1, "Stage1(Easy)", args.batch, args.imgsz, args.device, args.workers)

    print("  Stage2: Easy+Medium (从 Stage1 best 继续)")
    best_s2 = _train_one(best_s1, EM_YAML, args.stage_epochs,
                         "curriculum_stage2_em", args.batch, args.imgsz,
                         args.device, args.workers)
    s2 = evaluate(best_s2, "Stage2(Easy+Medium)", args.batch, args.imgsz, args.device, args.workers)

    print("  Stage3: 全量 (从 Stage2 best 继续)")
    best_s3 = _train_one(best_s2, ALL_YAML, args.stage_epochs,
                         "curriculum_stage3_all", args.batch, args.imgsz,
                         args.device, args.workers)
    s3 = evaluate(best_s3, "Stage3(All)", args.batch, args.imgsz, args.device, args.workers)
    return s1, s2, s3


def summarize(mixed, s1, s2, s3, base_dir, tag=''):
    lines = []
    lines.append("=" * 70)
    lines.append(f"VisDrone 实验最终结果  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(f"{'配置':<24}{'mAP50':>10}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}")
    lines.append("-" * 70)
    for m in [mixed, s1, s2, s3]:
        lines.append(f"{m['label']:<24}{m['mAP50']:>10}{m['mAP50-95']:>12}"
                     f"{m['Precision']:>12}{m['Recall']:>10}")
    gain = s3['mAP50'] - mixed['mAP50']
    lines.append(f"\n课程学习 (Stage3) 相对 Mixed 的 mAP50 增益: {gain:+.4f}")
    fname = f"visdrone_results_{tag}.txt" if tag else "visdrone_results.txt"
    out = os.path.join(base_dir, fname)
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)
    print(f"\n结果已保存至: {out}")


def main():
    global PROJECT, SPLIT, EASY_YAML, EM_YAML, ALL_YAML
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'mixed', 'curriculum'])
    parser.add_argument('--stage-epochs', type=int, default=40)
    parser.add_argument('--mixed-epochs', type=int, default=120)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)  # 默认 0, 避免 worker 进程内存峰值
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--tag', type=str, default='')  # 区分不同 epoch 设置的结果目录
    parser.add_argument('--split', type=str, default='VisDrone_split')  # 分组目录
    args = parser.parse_args()

    if args.split != 'VisDrone_split':
        SPLIT = os.path.join(BASE, args.split)
        EASY_YAML = os.path.join(SPLIT, "Easy", "data.yaml")
        EM_YAML = os.path.join(SPLIT, "EasyMedium", "data.yaml")
        ALL_YAML = os.path.join(SPLIT, "All", "data.yaml")

    if args.tag:
        PROJECT = os.path.join(BASE, f"runs_visdrone_{args.tag}")

    print(f"stage={args.stage}, stage-epochs={args.stage_epochs}, "
          f"mixed-epochs={args.mixed_epochs}, batch={args.batch}, workers={args.workers}, "
          f"tag={args.tag or '默认'}", flush=True)

    mixed = None
    s1 = s2 = s3 = None

    if args.stage in ('all', 'mixed'):
        mixed = run_mixed(args)
        print(f"  Mixed -> {mixed}", flush=True)

    if args.stage in ('all', 'curriculum'):
        s1, s2, s3 = run_curriculum(args)
        if args.stage == 'curriculum':
            # curriculum-only 时也评估本版本 Mixed, 便于汇总对比
            mixed_best = os.path.join(PROJECT, "mixed", "weights", "best.pt")
            if os.path.exists(mixed_best):
                mixed = evaluate(mixed_best, "Mixed", args.batch, args.imgsz,
                                 args.device, args.workers)
                print(f"  Mixed -> {mixed}", flush=True)

    if mixed and s1:
        summarize(mixed, s1, s2, s3, BASE, args.tag)


if __name__ == "__main__":
    main()
