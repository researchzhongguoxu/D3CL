"""
自动汇总 Pest24 40/50/60 epoch 课程学习结果
=====================================
- 等待 60ep 训练完成 (pest24_results_ep60.txt 出现)
- 解析 40/50/60 三个结果文件
- 生成最终对比表 pest24_final_summary.txt
- 三个 budget 共用同一 Mixed 基线 (120ep 饱和点 T=120)

用法:
  nohup python "5_汇总结果.py" > 汇总.log 2>&1 &
"""
import os
import time
from datetime import datetime

BASE = r"E:\dataset_create\Pest24"

FILES = {
    'ep40': os.path.join(BASE, 'pest24_results.txt'),
    'ep50': os.path.join(BASE, 'pest24_results_ep50.txt'),
    'ep60': os.path.join(BASE, 'pest24_results_ep60.txt'),
}
OUT = os.path.join(BASE, 'pest24_final_summary.txt')

CONFIGS = ['Mixed', 'Stage1(Easy)', 'Stage2(Easy+Medium)', 'Stage3(All)']


def parse_result(path):
    """解析结果文件, 返回 {配置名: {mAP50, mAP50-95, P, R}}"""
    if not os.path.exists(path):
        return None
    result = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] in CONFIGS:
                try:
                    result[parts[0]] = {
                        'mAP50': float(parts[1]),
                        'mAP50-95': float(parts[2]),
                        'P': float(parts[3]),
                        'R': float(parts[4]),
                    }
                except ValueError:
                    pass
    return result if result else None


def wait_for_completion(timeout_hours=6):
    """等待 60ep 结果文件出现; 超时返回 False"""
    deadline = time.time() + timeout_hours * 3600
    while not os.path.exists(FILES['ep60']):
        if time.time() > deadline:
            return False
        time.sleep(60)
    time.sleep(15)  # 确保文件写完整
    return True


def build_summary(results):
    lines = []
    lines.append("=" * 72)
    lines.append("Pest24 真实数据集实验汇总 (40/50/60 epoch budget)")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # Mixed 固定值
    mixed_mAP50 = None
    for tag in ['ep40', 'ep50', 'ep60']:
        r = results.get(tag)
        if r and 'Mixed' in r:
            mixed_mAP50 = r['Mixed']['mAP50']
            break
    if mixed_mAP50 is None:
        mixed_mAP50 = 0.2879  # 兜底用已知值
    lines.append(f"\nMixed 基线 (固定, 120ep 饱和): mAP50 = {mixed_mAP50:.4f}")
    lines.append("")

    # 三个 budget 的 Stage3 (课程学习最终) 对比
    lines.append("各 epoch budget 下课程学习最终 Stage3(All) 的 mAP50:")
    lines.append("-" * 72)
    lines.append(f"{'budget':<8}{'Stage3 mAP50':>14}{'vs Mixed 增益':>14}")
    lines.append("-" * 72)
    for tag in ['ep40', 'ep50', 'ep60']:
        r = results.get(tag)
        if r and 'Stage3(All)' in r:
            s3 = r['Stage3(All)']['mAP50']
            gain = s3 - mixed_mAP50
            lines.append(f"{tag:<8}{s3:>14.4f}{gain:>+14.4f}")
        else:
            lines.append(f"{tag:<8}{'未完成/缺失':>14}{'':>14}")
    lines.append("-" * 72)

    # 完整对比表 (mAP50)
    lines.append("\n完整 mAP50 对比:")
    lines.append("-" * 72)
    header = f"{'配置':<22}"
    for tag in ['ep40', 'ep50', 'ep60']:
        header += f"{tag:>10}"
    lines.append(header)
    lines.append("-" * 72)
    for cfg in CONFIGS:
        row = f"{cfg:<22}"
        for tag in ['ep40', 'ep50', 'ep60']:
            r = results.get(tag)
            if r and cfg in r:
                row += f"{r[cfg]['mAP50']:>10.4f}"
            else:
                row += f"{'-':>10}"
        lines.append(row)
    lines.append("-" * 72)

    # 完整指标表 (Stage3 的 P/R/mAP50-95)
    lines.append("\n课程学习 Stage3(All) 完整指标:")
    lines.append("-" * 72)
    lines.append(f"{'budget':<8}{'mAP50':>10}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}")
    lines.append("-" * 72)
    for tag in ['ep40', 'ep50', 'ep60']:
        r = results.get(tag)
        if r and 'Stage3(All)' in r:
            s3 = r['Stage3(All)']
            lines.append(f"{tag:<8}{s3['mAP50']:>10.4f}{s3['mAP50-95']:>12.4f}"
                         f"{s3['P']:>12.4f}{s3['R']:>10.4f}")
        else:
            lines.append(f"{tag:<8}{'未完成/缺失':>10}")
    lines.append("-" * 72)
    lines.append("\n注: Mixed 基线固定训练 120ep (饱和点 T=120), 三个 budget (T/3=40, T/3+10=50, T/3+20=60) 共用同一 Mixed 结果; batch=8 全程一致。")
    return "\n".join(lines)


def main():
    print(f"等待 60ep 训练完成... ({FILES['ep60']})", flush=True)
    ok = wait_for_completion(timeout_hours=6)
    if not ok:
        print("警告: 等待 60ep 超时, 用已有结果生成汇总。", flush=True)

    results = {}
    for tag, path in FILES.items():
        results[tag] = parse_result(path)
        status = "完成" if results[tag] else "缺失"
        print(f"  {tag}: {status}", flush=True)

    summary = build_summary(results)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(summary)
    print("\n" + summary, flush=True)
    print(f"\n最终汇总已保存至: {OUT}", flush=True)


if __name__ == '__main__':
    main()
