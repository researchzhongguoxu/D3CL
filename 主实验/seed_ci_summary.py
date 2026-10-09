# -*- coding: utf-8 -*-
"""
主实验种子置信区间 · 汇总
================================
读所有 (method, seed) 结果，计算 mean ± std 和 95% CI（n=5 → t=2.776），
输出到 seed_ci_summary.txt 和 seed_ci_summary.json。
"""
import os
import sys
import json
import math

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = r"E:\dataset_create"
RESULTS = os.path.join(BASE, "seed_ci_results")
OUT_TXT = os.path.join(BASE, "seed_ci_summary.txt")
OUT_JSON = os.path.join(BASE, "seed_ci_summary.json")

METHODS = ["Mixed", "Balanced", "D1", "D3"]
SEEDS = [42, 43, 44, 45, 46]
T95 = 2.776  # n=5, df=4, 双侧 95%


def ci95(vals):
    n = len(vals)
    if n < 2:
        return None
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    s = math.sqrt(var)
    half = T95 * s / math.sqrt(n)
    return m, s, (m - half, m + half)


def main():
    rows = []
    print("=" * 78, flush=True)
    print("主实验种子置信区间（5 seed, 95% CI, t=2.776）", flush=True)
    print("=" * 78, flush=True)

    for method in METHODS:
        vals = []
        missing = []
        for seed in SEEDS:
            p = os.path.join(RESULTS, f"{method}_seed{seed}.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                vals.append(d["mAP50"])
            else:
                missing.append(seed)

        line = f"{method:10s}"
        if vals:
            c = ci95(vals)
            m, s, (lo, hi) = c
            line += f"  n={len(vals)}  mean={m:.5f}  std={s:.5f}  95%CI=[{lo:.5f}, {hi:.5f}]"
            row = {"method": method, "n": len(vals), "mean": round(m, 5),
                   "std": round(s, 5), "ci_low": round(lo, 5), "ci_high": round(hi, 5),
                   "values": vals}
        else:
            line += "  无结果"
            row = {"method": method, "n": 0, "values": []}
        if missing:
            line += f"  (缺 seed: {missing})"
        rows.append(row)
        print(line, flush=True)

    # 两两比较 CI 是否重叠
    print("-" * 78, flush=True)
    print("两两 95% CI 重叠检查（重叠 = 差异不显著）:", flush=True)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["n"] >= 2 and b["n"] >= 2:
                overlap = a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]
                print(f"  {a['method']} vs {b['method']}: "
                      f"{'重叠(不显著)' if overlap else '不重叠(显著)'}", flush=True)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for r in rows:
            if r["n"] >= 2:
                f.write(f"{r['method']}\tn={r['n']}\tmean={r['mean']:.5f}\t"
                        f"std={r['std']:.5f}\tCI=[{r['ci_low']:.5f},{r['ci_high']:.5f}]\n")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {OUT_TXT} 和 {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
