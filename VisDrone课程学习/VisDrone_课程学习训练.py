"""
课程学习训练 (加速版 + 断点续传，一次跑完三阶段)
=====================================
- 全量数据分组: Easy(2157) / EasyMedium(4314) / All(6471)
- 三阶段: Stage1 Easy -> Stage2 EM -> Stage3 All，各阶段 epochs 可独立设置
- 每个 stage 训练完保存: add-{tag}-E.pt / -E+M.pt / -E+M+H.pt
- 断点续传: 每个 stage 检测 last.pt, 若中断过则 resume 从断点继续
- 加速: batch=8 + workers=4 + cache=disk
- 一次运行自动跑完三个 stage，无需中途手动操作

用法 (在 PyCharm 终端运行, Ctrl+C 可随时暂停, 重跑自动从断点续传):
  python "add_课程学习.py" --stage-epochs 15,30,200 2>&1 | tee "课程学习15_30_200.log"
"""
import os
import csv
import argparse
import shutil
from datetime import datetime
from ultralytics import YOLO

BASE = r"E:\dataset_create"
SPLIT = r"E:\dataset_create\visdrone\VisDrone_split"  # 全量分组
PROJECT = r"E:\dataset_create\visdrone\runs_curriculum"

EASY_YAML = os.path.join(SPLIT, "Easy", "data.yaml")
EM_YAML = os.path.join(SPLIT, "EasyMedium", "data.yaml")
ALL_YAML = os.path.join(SPLIT, "All", "data.yaml")


def stage_done(name, epochs):
    """检查某 stage 是否已完成 (results.csv 到 epochs 行)"""
    csv_path = os.path.join(PROJECT, name, "results.csv")
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return len(rows) >= epochs
    except Exception:
        return False


def train_stage(model_path, data_yaml, epochs, name):
    """训练一个阶段, 支持断点续传。返回 best.pt 路径"""
    best_pt = os.path.join(PROJECT, name, "weights", "best.pt")
    last_pt = os.path.join(PROJECT, name, "weights", "last.pt")

    # 1. 已完成则直接返回
    if stage_done(name, epochs) and os.path.exists(best_pt):
        print(f"  [{name}] 已完成, 跳过训练", flush=True)
        return best_pt

    common = dict(
        data=data_yaml,
        imgsz=640,
        batch=8,
        workers=4,
        cache='disk',
        device=0,
        project=PROJECT,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        plots=False,
        val=True,
        amp=True,
    )

    # 2. 中断过则 resume 续传
    if os.path.exists(last_pt):
        print(f"  [{name}] 检测到断点, 从 last.pt 续传...", flush=True)
        model = YOLO(last_pt)
        model.train(epochs=epochs, resume=True, **common)
    else:
        print(f"  [{name}] 从头训练...", flush=True)
        model = YOLO(model_path)
        model.train(epochs=epochs, **common)

    return best_pt


def run_curriculum(e1, e2, e3):
    tag = f"{e1}_{e2}_{e3}"
    print("=" * 70, flush=True)
    print(f"课程学习  三阶段 epochs: Stage1={e1} / Stage2={e2} / Stage3={e3}", flush=True)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    # Stage1: Easy
    print(f"\n[Stage1] Easy (2157 张, {e1} epochs)", flush=True)
    best1 = train_stage(os.path.join(BASE, "yolov8n.pt"), EASY_YAML, e1,
                        f"cl{tag}_s1_easy")
    dst1 = os.path.join(BASE, f"add-{tag}-E.pt")
    if os.path.exists(best1):
        shutil.copy2(best1, dst1)
        print(f"  [Stage1 保存] {os.path.basename(dst1)}", flush=True)

    # Stage2: Easy+Medium
    print(f"\n[Stage2] Easy+Medium (4314 张, {e2} epochs)", flush=True)
    best2 = train_stage(best1, EM_YAML, e2, f"cl{tag}_s2_em")
    dst2 = os.path.join(BASE, f"add-{tag}-E+M.pt")
    if os.path.exists(best2):
        shutil.copy2(best2, dst2)
        print(f"  [Stage2 保存] {os.path.basename(dst2)}", flush=True)

    # Stage3: All
    print(f"\n[Stage3] All (6471 张, {e3} epochs)", flush=True)
    best3 = train_stage(best2, ALL_YAML, e3, f"cl{tag}_s3_all")
    dst3 = os.path.join(BASE, f"add-{tag}-E+M+H.pt")
    if os.path.exists(best3):
        shutil.copy2(best3, dst3)
        print(f"  [Stage3 保存] {os.path.basename(dst3)}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"课程学习完成! Stage1={e1} / Stage2={e2} / Stage3={e3}", flush=True)
    print(f"三个模型已保存: add-{tag}-E.pt / add-{tag}-E+M.pt / add-{tag}-E+M+H.pt", flush=True)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-epochs', type=str, default='15,30,200',
                        help='三阶段 epochs，逗号分隔，如 15,30,200')
    args = parser.parse_args()
    e1, e2, e3 = [int(x) for x in args.stage_epochs.split(',')]
    run_curriculum(e1, e2, e3)
