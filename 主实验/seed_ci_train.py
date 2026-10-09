# -*- coding: utf-8 -*-
"""
主实验种子置信区间 · 单次训练（严格对齐论文）
============================================
对 Balanced / D1 / D3 某一方法 + 某一训练 seed，跑完整三阶段课程学习并评估。

- 数据：直接用 D 盘原始数据（论文训练时的原始图，动态 letterbox），
  data.yaml 在运行时动态生成（path 指向 D 盘，nc=1/names=mosquito）。
- 训练配置：逐项对齐论文 A-111-60-E+M+H/args.yaml，仅 seed 由 0 改为 42-46。
  optimizer=AdamW, lr0=0.002, momentum=0.937, weight_decay=0.0005,
  warmup_bias_lr=0.0, batch=4, nbs=64, deterministic=True, patience=30。
  cache 由论文的 ram 改为 disk（内存 3.8G 不够 ram，cache 不影响结果，像素一致）。
- 课程学习：三阶段串行（Stage1 Easy → Stage2 Easy+Medium → Stage3 All），
  每阶段用上一阶段 best.pt 初始化，每阶段相同 epoch。
- 最终评估：统一用 all/test（300 张）。

用法:
  python seed_ci_train.py --method Balanced --seed 42
"""
import os
import sys
import json
import argparse
import random
import yaml

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
from ultralytics import YOLO

BASE = r"E:\dataset_create"
D = r"D:/DDLC项目相关资料"
RUNS = os.path.join(BASE, "seed_ci_runs")
RESULTS = os.path.join(BASE, "seed_ci_results")
YAML_DIR = os.path.join(BASE, "seed_ci_yaml")

# 各阶段数据目录（D 盘原始数据）
DATA_DIRS = {
    "all":           "mix_all_three_safe",
    "balanced_easy": "separated_datasets/Easy",
    "balanced_em":   "111mix_easy_medium_safe",
    "d1_easy":       "212sorted_datasets_safe/Easy",
    "d1_em":         "212mix_easy_medium_safe",
    "d3_easy":       "121sorted_datasets_safe/Easy",
    "d3_em":         "121mix_easy_medium_safe",
}

# 方法与配置（每 stage 相同 epoch，对齐论文）
METHODS = {
    "Mixed":    {"epochs": 300, "stages": ["all"]},
    "Balanced": {"epochs": 60, "stages": ["balanced_easy", "balanced_em", "all"]},
    "D1":       {"epochs": 40, "stages": ["d1_easy", "d1_em", "all"]},
    "D3":       {"epochs": 60, "stages": ["d3_easy", "d3_em", "all"]},
}


def make_data_yaml(key):
    """生成指向 D 盘原始数据的 data.yaml（每次运行确保 path 正确）。"""
    root = os.path.join(D, DATA_DIRS[key]).replace("\\", "/")
    yaml_path = os.path.join(YAML_DIR, f"{key}.yaml")
    d = {
        "path": root,
        "train": f"{root}/images/train",
        "val": f"{root}/images/val",
        "test": f"{root}/images/test",
        "nc": 1,
        "names": ["mosquito"],
    }
    os.makedirs(YAML_DIR, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def make_test_yaml():
    """统一评估 yaml：val/test 均指向 all/test（300 张）。"""
    yaml_path = os.path.join(YAML_DIR, "test.yaml")
    root = os.path.join(D, DATA_DIRS["all"]).replace("\\", "/")
    d = {
        "path": root,
        "train": f"{root}/images/train",
        "val": f"{root}/images/test",
        "test": f"{root}/images/test",
        "nc": 1,
        "names": ["mosquito"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def train(method, seed):
    tag = f"{method}_seed{seed}"
    result_path = os.path.join(RESULTS, f"{tag}.json")
    if os.path.exists(result_path):
        print(f"[跳过] {tag} 已有结果", flush=True)
        return True

    cfg = METHODS[method]
    stages = cfg["stages"]
    ep = cfg["epochs"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    common = dict(
        imgsz=640,
        batch=4,
        workers=2,
        cache="disk",
        device=0,
        project=RUNS,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.002,
        momentum=0.937,
        warmup_bias_lr=0.0,
        deterministic=True,
        patience=30,
        amp=True,
        verbose=True,
        plots=False,
        val=True,
    )

    print(f"[开始] {tag}  epochs={ep}  stages={stages}", flush=True)

    model_path = os.path.join(BASE, "yolov8n.pt")
    best_pt = None
    try:
        for i, dk in enumerate(stages):
            stage_name = f"{tag}_s{i + 1}"
            stage_best = os.path.join(RUNS, stage_name, "weights", "best.pt")
            if os.path.exists(stage_best):
                print(f"  [{tag}] Stage{i + 1}({dk}) 已有 best.pt，跳过", flush=True)
                best_pt = stage_best
                model_path = stage_best
                continue
            stage_yaml = make_data_yaml(dk)
            model = YOLO(model_path)
            model.train(data=stage_yaml, epochs=ep, seed=seed, name=stage_name, **common)
            if os.path.exists(stage_best):
                best_pt = stage_best
                model_path = stage_best
                print(f"  [{tag}] Stage{i + 1}({dk}) 完成 -> {stage_best}", flush=True)
            else:
                print(f"  [{tag}] Stage{i + 1}({dk}) 训练后未找到 best.pt，异常！", flush=True)
                return False

        test_yaml = make_test_yaml()
        model = YOLO(best_pt)
        metrics = model.val(data=test_yaml, imgsz=640, batch=16, device=0,
                            workers=2, verbose=False, plots=False)
        rd = metrics.results_dict
        result = {
            "method": method,
            "seed": seed,
            "mAP50": round(float(rd["metrics/mAP50(B)"]), 5),
            "mAP50_95": round(float(rd["metrics/mAP50-95(B)"]), 5),
            "P": round(float(rd["metrics/precision(B)"]), 5),
            "R": round(float(rd["metrics/recall(B)"]), 5),
        }
        os.makedirs(RESULTS, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[完成] {tag}: mAP50={result['mAP50']}  mAP50-95={result['mAP50_95']}", flush=True)
        return True
    except Exception as e:
        print(f"[失败] {tag}: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    ok = train(args.method, args.seed)
    sys.exit(0 if ok else 1)
