import os
import shutil
import random
import yaml
import warnings

warnings.filterwarnings('ignore')


def collect_triplets_from_split(source_dirs, split):
    triplets = []

    for source_dir in source_dirs:
        images_dir = os.path.join(source_dir, "images", split)
        labels_dir = os.path.join(source_dir, "labels", split)
        ods_dir = os.path.join(source_dir, "ods_values", split)

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir) or not os.path.exists(ods_dir):
            continue

        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(images_dir, filename)
                base_name = os.path.splitext(filename)[0]

                lbl_path = os.path.join(labels_dir, base_name + '.txt')
                ods_path = os.path.join(ods_dir, base_name + '.ods')

                if not os.path.exists(lbl_path) or not os.path.exists(ods_path):
                    continue

                try:
                    with open(ods_path, 'r', encoding='utf-8') as f:
                        ods_val = float(f.read().strip())
                except Exception:
                    continue

                triplets.append((img_path, lbl_path, ods_val))

    return triplets


def save_triplets_to_split(triplets, split_name, output_merged_dir):
    output_img_dir = os.path.join(output_merged_dir, "images", split_name)
    output_lbl_dir = os.path.join(output_merged_dir, "labels", split_name)
    output_ods_dir = os.path.join(output_merged_dir, "ods_values", split_name)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)
    os.makedirs(output_ods_dir, exist_ok=True)

    for i, (src_img, src_lbl, ods_val) in enumerate(triplets):
        src_ext = os.path.splitext(src_img)[1]
        new_base_name = f"merged_{split_name}_{i + 1:06d}"

        shutil.copy2(src_img, os.path.join(output_img_dir, f"{new_base_name}{src_ext}"))
        shutil.copy2(src_lbl, os.path.join(output_lbl_dir, f"{new_base_name}.txt"))

        with open(os.path.join(output_ods_dir, f"{new_base_name}.ods"), 'w', encoding='utf-8') as f:
            f.write(f"{ods_val:.6f}")


def merge_easy_medium_without_leakage(input_root_dir, output_merged_dir):
    medium_dir = os.path.join(input_root_dir, "Medium")
    easy_dir = os.path.join(input_root_dir, "Easy")
    source_dirs = [medium_dir, easy_dir]

    train_triplets = collect_triplets_from_split(source_dirs, "train")
    test_triplets = collect_triplets_from_split(source_dirs, "test")
    val_triplets = collect_triplets_from_split(source_dirs, "val")

    random.shuffle(train_triplets)
    random.shuffle(test_triplets)
    random.shuffle(val_triplets)

    save_triplets_to_split(train_triplets, "train", output_merged_dir)
    save_triplets_to_split(test_triplets, "test", output_merged_dir)
    save_triplets_to_split(val_triplets, "val", output_merged_dir)

    data_yaml_path = os.path.join(output_merged_dir, "data.yaml")
    level_path = output_merged_dir.replace('\\', '/')

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

    print("完成：Easy + Medium 已安全混合。")


if __name__ == "__main__":
    INPUT_ROOT_DIR = r"C:\Users\wgl20\Desktop\532sorted_datasets_safe"
    MERGED_OUTPUT_PATH = r"C:\Users\wgl20\Desktop\532mix_easy_medium_safe"

    merge_easy_medium_without_leakage(INPUT_ROOT_DIR, MERGED_OUTPUT_PATH)