# -*- coding: utf-8 -*-
"""
主实验种子置信区间 · 主控脚本
================================
串行跑 3 方法 × 5 seed = 15 次训练，然后汇总。
- 断点续传：已有结果的 (method, seed) 自动跳过。
- 单次失败不中断：记录日志后继续下一个。
- 用 nohup 启动后可脱离终端持续运行。

用法:
  nohup python seed_ci_run_all.py > seed_ci_run_all.log 2>&1 &
"""
import os
import sys
import subprocess
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = r"E:\dataset_create"
RESULTS = os.path.join(BASE, "seed_ci_results")

METHODS = ["Mixed", "Balanced", "D1", "D3"]
SEEDS = [42, 43, 44, 45, 46]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 60)
    log("主实验种子置信区间 开始")
    log(f"方法: {METHODS}")
    log(f"种子: {SEEDS}")
    log(f"共 {len(METHODS) * len(SEEDS)} 次训练")
    log("=" * 60)

    total = 0
    done = 0
    failed = 0
    for method in METHODS:
        for seed in SEEDS:
            total += 1
            tag = f"{method}_seed{seed}"
            result_path = os.path.join(RESULTS, f"{tag}.json")
            if os.path.exists(result_path):
                log(f"[跳过] {tag} 已有结果")
                done += 1
                continue
            log(f"[训练] {tag}")
            cmd = [sys.executable, os.path.join(BASE, "seed_ci_train.py"),
                   "--method", method, "--seed", str(seed)]
            ret = subprocess.run(cmd, cwd=BASE)
            if ret.returncode == 0:
                done += 1
                log(f"[完成] {tag}")
            else:
                failed += 1
                log(f"[失败] {tag} returncode={ret.returncode}")

    log("-" * 60)
    log(f"训练结束: 总 {total} / 完成 {done} / 失败 {failed}")

    # 汇总
    log("开始汇总...")
    subprocess.run([sys.executable, os.path.join(BASE, "seed_ci_summary.py")], cwd=BASE)
    log("全部结束")


if __name__ == "__main__":
    main()
