import os
import shutil
import random
import yaml
import warnings

warnings.filterwarnings('ignore')


def collect_samples_from_split(data_dir, split):
    """
    只从指定 split(train/val/test) 收集样本，避免数据泄露。
    返回: [(img_path, lbl_path, ods_val), ...]
    """
    triplets = []

    images_dir = os.path.join(data_dir, "images", split)
    labels_dir = os.path.join(data_dir, "labels", split)
    ods_dir = os.path.join(data_dir, "ods_values", split)

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir) or not os.path.exists(ods_dir):
        print(f"警告: 缺少目录，跳过 {split}: {images_dir}")
        return triplets

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            img_path = os.path.join(images_dir, filename)
            base_name = os.path.splitext(filename)[0]

            lbl_path = os.path.join(labels_dir, base_name + '.txt')
            ods_path = os.path.join(ods_dir, base_name + '.ods')

            if not os.path.exists(lbl_path):
                print(f"警告: 未找到标签文件，跳过 {lbl_path}")
                continue
            if not os.path.exists(ods_path):
                print(f"警告: 未找到ODS文件，跳过 {ods_path}")
                continue

            try:
                with open(ods_path, 'r', encoding='utf-8') as f:
                    ods_val = float(f.read().strip())
                triplets.append((img_path, lbl_path, ods_val))
            except Exception as e:
                print(f"警告: 无法读取ODS文件 {ods_path}: {str(e)}")

    return triplets


def split_samples_by_ratio(samples, easy_ratio, medium_ratio, hard_ratio):
    """
    按 ODS 从小到大排序后，按比例切分为 Hard / Medium / Easy
    """
    samples = sorted(samples, key=lambda x: x[2])  # ODS小 -> 更难

    total = len(samples)
    hard_count = int(total * hard_ratio)
    medium_count = int(total * medium_ratio)
    easy_count = total - hard_count - medium_count  # 保证总和正确

    hard_samples = samples[:hard_count]
    medium_samples = samples[hard_count:hard_count + medium_count]
    easy_samples = samples[hard_count + medium_count:]

    return {
        "Hard": hard_samples,
        "Medium": medium_samples,
        "Easy": easy_samples
    }


def save_level_split(level_name, split_name, samples, output_dir):
    """
    把指定 level 的指定 split 样本保存到目标目录
    """
    img_dest_dir = os.path.join(output_dir, level_name, "images", split_name)
    lbl_dest_dir = os.path.join(output_dir, level_name, "labels", split_name)
    ods_dest_dir = os.path.join(output_dir, level_name, "ods_values", split_name)

    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(lbl_dest_dir, exist_ok=True)
    os.makedirs(ods_dest_dir, exist_ok=True)

    # 为了避免每个 level 内部还保留ODS顺序痕迹，保存前打乱
    samples = samples.copy()
    random.shuffle(samples)

    for i, (src_img, src_lbl, ods_val) in enumerate(samples):
        src_ext = os.path.splitext(src_img)[1]  # 保留原图片扩展名
        new_base_name = f"{level_name.lower()}_{split_name}_{i + 1:06d}"

        new_img_name = f"{new_base_name}{src_ext}"
        new_lbl_name = f"{new_base_name}.txt"
        new_ods_name = f"{new_base_name}.ods"

        shutil.copy2(src_img, os.path.join(img_dest_dir, new_img_name))
        shutil.copy2(src_lbl, os.path.join(lbl_dest_dir, new_lbl_name))

        with open(os.path.join(ods_dest_dir, new_ods_name), 'w', encoding='utf-8') as f:
            f.write(f"{ods_val:.6f}")


def create_sorted_datasets_no_leakage(input_dir, output_dir, easy_ratio=0.5, medium_ratio=0.3, hard_ratio=0.2):
    """
    从输入数据集中按 ODS 排序，按比例分成 Easy / Medium / Hard 三个新数据集。
    关键点：train / val / test 分开处理，不会发生数据泄露。
    """

    # 1. 检查比例是否合法
    ratio_sum = easy_ratio + medium_ratio + hard_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为 1，目前为 {ratio_sum}")

    # 2. 分别收集 train / val / test（严格分开）
    train_samples = collect_samples_from_split(input_dir, "train")
    val_samples = collect_samples_from_split(input_dir, "val")
    test_samples = collect_samples_from_split(input_dir, "test")

    if not train_samples and not val_samples and not test_samples:
        print("错误: 未找到任何有效样本。")
        return

    print(f"共找到 train 样本: {len(train_samples)} 个")
    print(f"共找到 val 样本: {len(val_samples)} 个")
    print(f"共找到 test 样本: {len(test_samples)} 个")

    # 3. 分别按 ODS 排序并按比例分桶
    train_split = split_samples_by_ratio(train_samples, easy_ratio, medium_ratio, hard_ratio)
    val_split = split_samples_by_ratio(val_samples, easy_ratio, medium_ratio, hard_ratio)
    test_split = split_samples_by_ratio(test_samples, easy_ratio, medium_ratio, hard_ratio)

    # 4. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    levels = ["Hard", "Medium", "Easy"]

    # 5. 保存 train / val / test（分别处理，不重新划分）
    for level in levels:
        save_level_split(level, "train", train_split[level], output_dir)
        save_level_split(level, "val", val_split[level], output_dir)
        save_level_split(level, "test", test_split[level], output_dir)

    # 6. 打印统计信息
    print("数据集已按ODS排序并重新分桶（无数据泄露）:")
    for level in levels:
        train_count = len(train_split[level])
        val_count = len(val_split[level])
        test_count = len(test_split[level])
        total_count = train_count + val_count + test_count

        if level == "Hard":
            ratio = hard_ratio
            desc = "ODS值最小，最难检测"
        elif level == "Medium":
            ratio = medium_ratio
            desc = "中等难度"
        else:
            ratio = easy_ratio
            desc = "ODS值最大，最容易检测"

        print(
            f"  {level}: {total_count} 张 ({ratio:.0%}), "
            f"其中 train={train_count}, val={val_count}, test={test_count} ({desc})"
        )

    # 7. 为每个 level 生成 data.yaml
    for level in levels:
        data_yaml_path = os.path.join(output_dir, level, "data.yaml")
        level_path = os.path.join(output_dir, level).replace('\\', '/')

        yaml_data = {
            'path': level_path,
            'train': f'{level_path}/images/train',
            'val': f'{level_path}/images/val',
            'test': f'{level_path}/images/test',
            'nc': 1,
            'names': ['mosquito']
        }

        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"  {level} 级别的YAML文件已保存至: {data_yaml_path}")


def main():
    input_dataset = r"C:\Users\wgl20\Desktop\mix_all_three_safe"
    output_dataset = r"C:\Users\wgl20\Desktop\532sorted_datasets_safe"

    easy_ratio = 0.5
    medium_ratio = 0.3
    hard_ratio = 0.2

    print(f"正在处理数据集: {input_dataset}")
    print(f"输出目录: {output_dataset}")

    create_sorted_datasets_no_leakage(
        input_dataset,
        output_dataset,
        easy_ratio,
        medium_ratio,
        hard_ratio
    )

    print("\n处理完成！")
    print(f"新数据集已保存至: {output_dataset}")
    print("注意：原 train 只参与新 train，原 val 只参与新 val，原 test 只参与新 test，因此不会发生数据泄露。")


if __name__ == "__main__":
    main()