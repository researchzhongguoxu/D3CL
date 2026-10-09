import os
import cv2
import numpy as np
import random
import yaml
import warnings
import shutil
from albumentations import (
    Compose, RandomBrightnessContrast, OneOf,
    GaussianBlur, MotionBlur, MedianBlur,
    RGBShift, ChannelShuffle, ImageCompression,
    ToFloat, FromFloat
)

warnings.filterwarnings("ignore")

# =========================
# 0. 固定随机种子
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
cv2.setRNGSeed(SEED)

# =========================
# 1. 配置参数
# =========================
BACKGROUND_DIR = r"C:\Users\wgl20\Desktop\background"
MOSQUITO_DIR = r"C:\Users\wgl20\Desktop\filtered_mosquito_processed"
OUTPUT_BASE_DIR = r"C:\Users\wgl20\Desktop\separated_datasets"

TOTAL_SAMPLES = 1500

# 7:2:1
TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1

EPSILON = 1e-6

# =========================
# 2. 更偏“可学”的生成参数
# =========================
SIZE_RANGE = (0.12, 0.55)
CONTRAST_RANGE = (-0.22, 0.22)
BRIGHTNESS_RANGE = (-0.18, 0.18)
BLUR_PROB_RANGE = (0.05, 0.55)
COLOR_PROB_RANGE = (0.10, 0.45)
ISO_NOISE_PROB_RANGE = (0.08, 0.40)


# =========================
# 3. 工具函数
# =========================
def get_all_images_from_dir(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def cycle(iterable):
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
            yield element


def safe_imwrite(path, image):
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Failed to save image to: {path}")


def normalize_to_unit_interval(values):
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < EPSILON:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin + EPSILON) for v in values]


def find_largest_contour_bbox(mask):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h


def calculate_sharpness_laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(sharpness)


def calculate_local_contrast(mask, obj_region, bg_region):
    """
    C = |mu_obj - mu_bg| / (sigma_bg + epsilon)
    在 LAB 的 L 通道上计算
    """
    obj_lab = cv2.cvtColor(obj_region, cv2.COLOR_BGR2LAB)
    bg_lab = cv2.cvtColor(bg_region, cv2.COLOR_BGR2LAB)

    mean_obj_L, _ = cv2.meanStdDev(obj_lab[:, :, 0], mask=mask.astype(np.uint8))
    mean_obj_L = mean_obj_L[0][0]

    mean_bg_L, std_dev_bg_L = cv2.meanStdDev(bg_lab[:, :, 0])
    mean_bg_L = mean_bg_L[0][0]
    std_dev_bg_L = std_dev_bg_L[0][0]

    C = abs(mean_obj_L - mean_bg_L) / (std_dev_bg_L + EPSILON)
    return float(C)


def apply_iso_noise(image, prob):
    """
    保留 ISO 噪声，但把强度控制得更温和
    """
    if random.random() > prob:
        return image

    img = image.astype(np.float32) / 255.0

    luminance_sigma = random.uniform(0.005, 0.03)
    chroma_sigma = random.uniform(0.003, 0.015)

    lum_noise = np.random.normal(0, luminance_sigma, img.shape[:2]).astype(np.float32)
    lum_noise = np.expand_dims(lum_noise, axis=2)

    color_noise = np.random.normal(0, chroma_sigma, img.shape).astype(np.float32)

    gain = np.sqrt(np.clip(img, 0.0, 1.0) + 1e-6)

    noisy = img + lum_noise * gain + color_noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def build_transform(brightness_val, contrast_val, blur_p, color_p):
    try:
        compression_tf = ImageCompression(quality_range=(82, 95), p=0.35)
    except TypeError:
        compression_tf = ImageCompression(quality_lower=82, quality_upper=95, p=0.35)

    transform = Compose([
        OneOf([
            GaussianBlur(blur_limit=(3, 7), p=0.6),
            MotionBlur(blur_limit=3, p=0.3),
            MedianBlur(blur_limit=3, p=0.1),
        ], p=blur_p),
        RandomBrightnessContrast(
            brightness_limit=brightness_val,
            contrast_limit=contrast_val,
            p=0.75
        ),
        OneOf([
            RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=color_p),
            ChannelShuffle(p=max(color_p * 0.35, 0.0)),
        ], p=color_p),
        compression_tf,
        ToFloat(max_value=255.0),
        FromFloat(dtype="uint8", max_value=255.0)
    ])
    return transform


def stratified_split(samples, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, num_bins=10):
    """
    在一个等级内部做分层切分，尽量让 train / test / val 的 ODS 分布更接近
    samples: 已经是某个等级内的样本列表，每个元素都含有 ODS
    """
    if len(samples) == 0:
        return [], [], []

    samples = sorted(samples, key=lambda x: x["ODS"])
    bins = np.array_split(samples, min(num_bins, len(samples)))

    train_split, test_split, val_split = [], [], []

    for bin_samples in bins:
        bin_samples = list(bin_samples)
        random.shuffle(bin_samples)

        n = len(bin_samples)
        n_train = int(round(n * train_ratio))
        n_test = int(round(n * test_ratio))

        # 修正边界，确保总数对齐
        if n_train + n_test > n:
            n_test = max(0, n - n_train)
        n_val = n - n_train - n_test

        train_split.extend(bin_samples[:n_train])
        test_split.extend(bin_samples[n_train:n_train + n_test])
        val_split.extend(bin_samples[n_train + n_test:])

    return train_split, test_split, val_split


# =========================
# 4. 单张样本生成
# =========================
def create_single_synthetic_sample(bg_img, mosquito_img, output_image_path, output_label_path):
    if bg_img is None or mosquito_img is None:
        return None

    if mosquito_img.shape[2] != 4:
        print("Warning: mosquito image does not have alpha channel. Skipping.")
        return None

    bg_h, bg_w = bg_img.shape[:2]
    canvas_size = max(bg_h, bg_w)

    scale_factor = random.uniform(*SIZE_RANGE)
    contrast_val = random.uniform(*CONTRAST_RANGE)
    brightness_val = random.uniform(*BRIGHTNESS_RANGE)
    blur_p = random.uniform(*BLUR_PROB_RANGE)
    color_p = random.uniform(*COLOR_PROB_RANGE)
    iso_p = random.uniform(*ISO_NOISE_PROB_RANGE)

    new_w = max(1, int(mosquito_img.shape[1] * scale_factor))
    new_h = max(1, int(mosquito_img.shape[0] * scale_factor))

    resized_rgb = cv2.resize(mosquito_img[:, :, :3], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(mosquito_img[:, :, 3], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    rotation_angle = random.uniform(-180, 180)
    rotation_matrix = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), rotation_angle, 1.0)

    rotated_rgb = cv2.warpAffine(
        resized_rgb,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    rotated_alpha = cv2.warpAffine(
        resized_alpha,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    target_mask_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    start_y = (canvas_size - bg_h) // 2
    start_x = (canvas_size - bg_w) // 2
    canvas[start_y:start_y + bg_h, start_x:start_x + bg_w] = bg_img

    max_x = start_x + bg_w - rotated_rgb.shape[1]
    max_y = start_y + bg_h - rotated_rgb.shape[0]
    if max_x < start_x or max_y < start_y:
        return None

    pos_x = random.randint(start_x, max_x)
    pos_y = random.randint(start_y, max_y)

    roi = canvas[pos_y:pos_y + rotated_rgb.shape[0], pos_x:pos_x + rotated_rgb.shape[1]]

    alpha = rotated_alpha.astype(np.float32) / 255.0
    alpha_3 = np.dstack([alpha, alpha, alpha])

    blended_roi = (1.0 - alpha_3) * roi.astype(np.float32) + alpha_3 * rotated_rgb.astype(np.float32)
    canvas[pos_y:pos_y + rotated_rgb.shape[0], pos_x:pos_x + rotated_rgb.shape[1]] = blended_roi.astype(np.uint8)

    target_region = (rotated_alpha > 0).astype(np.uint8) * 255
    target_mask_canvas[
        pos_y:pos_y + rotated_rgb.shape[0],
        pos_x:pos_x + rotated_rgb.shape[1]
    ] = np.maximum(
        target_mask_canvas[pos_y:pos_y + rotated_rgb.shape[0], pos_x:pos_x + rotated_rgb.shape[1]],
        target_region
    )

    final_img = canvas[start_y:start_y + bg_h, start_x:start_x + bg_w]
    final_mask = target_mask_canvas[start_y:start_y + bg_h, start_x:start_x + bg_w]

    bbox = find_largest_contour_bbox(final_mask)
    if bbox is None:
        return None

    transform = build_transform(brightness_val, contrast_val, blur_p, color_p)
    enhanced_result = transform(image=final_img, mask=final_mask)

    final_enhanced_img = enhanced_result["image"]
    final_enhanced_mask = enhanced_result["mask"].astype(np.uint8)

    final_enhanced_img = apply_iso_noise(final_enhanced_img, iso_p)

    enhanced_bbox = find_largest_contour_bbox(final_enhanced_mask)
    if enhanced_bbox is None:
        return None

    x, y, w, h = enhanced_bbox
    if w <= 0 or h <= 0:
        return None

    obj_mask = final_enhanced_mask[y:y + h, x:x + w]
    obj_region = final_enhanced_img[y:y + h, x:x + w]

    margin = int(max(w, h) * 0.2)
    x_bg_start = max(0, x - margin)
    y_bg_start = max(0, y - margin)
    x_bg_end = min(final_enhanced_img.shape[1], x + w + margin)
    y_bg_end = min(final_enhanced_img.shape[0], y + h + margin)

    bg_patch = final_enhanced_img[y_bg_start:y_bg_end, x_bg_start:x_bg_end]

    C = calculate_local_contrast(obj_mask, obj_region, bg_patch)
    S = float(np.sqrt(w * h))
    sharpness = calculate_sharpness_laplacian_variance(obj_region)

    safe_imwrite(output_image_path, final_enhanced_img)

    center_x = (x + w / 2) / final_enhanced_img.shape[1]
    center_y = (y + h / 2) / final_enhanced_img.shape[0]
    width = w / final_enhanced_img.shape[1]
    height = h / final_enhanced_img.shape[0]

    yolo_label = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
    with open(output_label_path, "w", encoding="utf-8") as f:
        f.write(yolo_label)

    return {
        "image_path": output_image_path,
        "label_path": output_label_path,
        "C": C,
        "S": S,
        "sharpness": sharpness
    }


# =========================
# 5. 主流程
# =========================
def main():
    print("正在加载背景和蚊子图像...")
    bg_images = get_all_images_from_dir(BACKGROUND_DIR)
    mosquito_images = get_all_images_from_dir(MOSQUITO_DIR)

    if not bg_images or not mosquito_images:
        print("错误：未找到背景或蚊子图像。")
        return

    # 先清空旧输出，避免残留文件影响分布
    if os.path.exists(OUTPUT_BASE_DIR):
        shutil.rmtree(OUTPUT_BASE_DIR)

    print(f"找到 {len(bg_images)} 张背景图和 {len(mosquito_images)} 张蚊子图。")

    random.shuffle(bg_images)
    random.shuffle(mosquito_images)

    print(f"\n--- 开始生成总计 {TOTAL_SAMPLES} 张图片 ---")

    temp_dir = os.path.join(OUTPUT_BASE_DIR, "temp_all_samples")
    temp_img_dir = os.path.join(temp_dir, "images")
    temp_lbl_dir = os.path.join(temp_dir, "labels")
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_lbl_dir, exist_ok=True)

    bg_cycle = cycle(bg_images)
    mosquito_cycle = cycle(mosquito_images)

    all_samples = []

    for i in range(TOTAL_SAMPLES):
        bg_path = next(bg_cycle)
        mosquito_path = next(mosquito_cycle)

        bg_img = cv2.imread(bg_path)
        mosquito_img = cv2.imread(mosquito_path, cv2.IMREAD_UNCHANGED)

        temp_img_filename = f"all_samples_{i:04d}.jpg"
        temp_lbl_filename = f"all_samples_{i:04d}.txt"

        temp_img_path = os.path.join(temp_img_dir, temp_img_filename)
        temp_lbl_path = os.path.join(temp_lbl_dir, temp_lbl_filename)

        sample_info = create_single_synthetic_sample(
            bg_img,
            mosquito_img,
            temp_img_path,
            temp_lbl_path
        )

        if sample_info is not None:
            all_samples.append(sample_info)

        if (i + 1) % 500 == 0:
            print(f"  已生成 {i + 1}/{TOTAL_SAMPLES} 张...")

    total_valid = len(all_samples)
    if total_valid == 0:
        print("错误：没有生成有效样本。")
        return

    print(f"成功生成 {total_valid} 张有效图片，开始计算 ODS 并排序...")

    # B 是 Blur，不是 Sharpness
    sharpness_values = [s["sharpness"] for s in all_samples]
    sharpness_norm = normalize_to_unit_interval(sharpness_values)
    blur_severity_values = [1.0 - v for v in sharpness_norm]

    for i, sample in enumerate(all_samples):
        B = blur_severity_values[i]
        sample["B"] = float(B)

    # ========== 论文式 1-4：C、S、B 全部 min-max 归一化后再算 ODS ==========
    # C_norm = (C-Cmin)/(Cmax-Cmin), S_norm = (S-Smin)/(Smax-Smin), B_norm = (B-Bmin)/(Bmax-Bmin)
    # ODS = C_norm * S_norm / (1 + B_norm)   (alpha = beta = gamma = 1)
    C_norm = normalize_to_unit_interval([s["C"] for s in all_samples])
    S_norm = normalize_to_unit_interval([s["S"] for s in all_samples])
    B_norm = normalize_to_unit_interval([s["B"] for s in all_samples])

    for i, sample in enumerate(all_samples):
        sample["ODS"] = float((C_norm[i] * S_norm[i]) / (1.0 + B_norm[i]))

    all_samples.sort(key=lambda x: x["ODS"])   # ODS小 -> 更难

    # ========== 诊断输出：公式阈值分析 ==========
    C_vals = [s["C"] for s in all_samples]
    S_vals = [s["S"] for s in all_samples]
    B_vals = [s["B"] for s in all_samples]
    ODS_vals = [s["ODS"] for s in all_samples]
    sharp_raw = [s["sharpness"] for s in all_samples]

    print("\n" + "="*70)
    print("========== 公式阈值诊断报告 ==========")
    print("="*70)

    # --- C (局部对比度) 统计 ---
    print("\n--- 1. 局部对比度 C = |μ_obj-μ_bg|/(σ_bg+ε) ---")
    print(f"  C raw:  min={min(C_vals):.4f},  max={max(C_vals):.4f},  mean={np.mean(C_vals):.4f},  median={np.median(C_vals):.4f}")

    # --- S (目标尺寸) 统计 ---
    print("\n--- 2. 目标尺寸 S = sqrt(w*h) ---")
    print(f"  S raw:  min={min(S_vals):.2f},  max={max(S_vals):.2f},  mean={np.mean(S_vals):.2f},  median={np.median(S_vals):.2f}")

    # --- B (模糊退化) 统计 ---
    print("\n--- 3. 模糊退化 B = 1 - normalized_sharpness ---")
    print(f"  B:      min={min(B_vals):.4f},  max={max(B_vals):.4f},  mean={np.mean(B_vals):.4f},  median={np.median(B_vals):.4f}")
    print(f"  sharpness raw: min={min(sharp_raw):.2f}, max={max(sharp_raw):.2f}")

    # --- 4. ODS 统计（论文式 1-4 归一化版） ---
    print("\n--- 4. ODS = C_norm * S_norm / (1 + B_norm)  [论文式1-4, C/S/B 全部归一化] ---")
    print(f"  ODS:    min={min(ODS_vals):.6f},  max={max(ODS_vals):.6f},  mean={np.mean(ODS_vals):.6f},  median={np.median(ODS_vals):.6f}")

    # --- 5. 归一化分量 C_norm / S_norm / B_norm 统计 ---
    print("\n--- 5. 归一化分量 C_norm / S_norm / B_norm ---")
    print(f"  C_norm: min={min(C_norm):.4f}, max={max(C_norm):.4f}")
    print(f"  S_norm: min={min(S_norm):.4f}, max={max(S_norm):.4f}")
    print(f"  B_norm: min={min(B_norm):.4f}, max={max(B_norm):.4f}")

    split_index_1 = total_valid // 3
    split_index_2 = 2 * total_valid // 3

    # --- 关键阈值 ---
    print("\n" + "="*70)
    print("========== 公式 10-12 难度级别阈值 ==========")
    print(f"  总样本数: {total_valid}")
    print(f"  q1 = 33%:  index={split_index_1}, ODS阈值 ≈ {all_samples[split_index_1]['ODS']:.6f}")
    print(f"  q2 = 67%:  index={split_index_2}, ODS阈值 ≈ {all_samples[split_index_2]['ODS']:.6f}")
    print(f"  (ODS 归一化后，q1/q2 为 ODS 升序的 33%/67% 分位点)")
    print()

    hard_samples = all_samples[:split_index_1]
    medium_samples = all_samples[split_index_1:split_index_2]
    easy_samples = all_samples[split_index_2:]

    for name, samples in [("Hard", hard_samples), ("Medium", medium_samples), ("Easy", easy_samples)]:
        o = [s["ODS"] for s in samples]
        print(f"  {name:8s}: ODS ∈ [{min(o):.4f}, {max(o):.4f}], mean={np.mean(o):.4f}")

    # --- 对应论文式 18 的 τ 区间（ODS 归一化，按升序取分位点） ---
    print("\n========== 公式 18 对应阈值 τ ==========")
    print(f"  τL_Hard = {ODS_vals[0]:.6f},  τU_Hard = {ODS_vals[split_index_1-1]:.6f}")
    print(f"  τL_Medium = {ODS_vals[split_index_1]:.6f},  τU_Medium = {ODS_vals[split_index_2-1]:.6f}")
    print(f"  τL_Easy = {ODS_vals[split_index_2]:.6f},  τU_Easy = {ODS_vals[-1]:.6f}")

    print("="*70 + "\n")

    levels_data = {
        "Hard": hard_samples,
        "Medium": medium_samples,
        "Easy": easy_samples
    }

    # --- 每个级别的 C, S, B 详细统计 ---
    print("\n========== 各级别 C, S, B 统计 ==========")
    for name, samples in [("Hard", hard_samples), ("Medium", medium_samples), ("Easy", easy_samples)]:
        c = [s["C"] for s in samples]
        s_ = [s["S"] for s in samples]
        b = [s["B"] for s in samples]
        sh = [s["sharpness"] for s in samples]
        ods = [s["ODS"] for s in samples]
        print(f"\n  [{name}] (n={len(samples)})")
        print(f"    C:  min={min(c):.4f}, max={max(c):.4f}, mean={np.mean(c):.4f}, median={np.median(c):.4f}")
        print(f"    S:  min={min(s_):.2f}, max={max(s_):.2f}, mean={np.mean(s_):.2f}, median={np.median(s_):.2f}")
        print(f"    B:  min={min(b):.4f}, max={max(b):.4f}, mean={np.mean(b):.4f}, median={np.median(b):.4f}")
        print(f"    sharpness: min={min(sh):.2f}, max={max(sh):.2f}, mean={np.mean(sh):.2f}")
        print(f"    ODS: min={min(ods):.4f}, max={max(ods):.4f}, mean={np.mean(ods):.4f}")
    print("="*70 + "\n")

    print("\n--- 自动分级完成 ---")
    print(f"Hard:   {len(hard_samples)}")
    print(f"Medium: {len(medium_samples)}")
    print(f"Easy:   {len(easy_samples)}")

    for level, samples_list in levels_data.items():
        print(f"\n--- 整理 {level} 级别数据 ---")

        level_dirs = {
            "train": {
                "img": os.path.join(OUTPUT_BASE_DIR, level, "images", "train"),
                "lbl": os.path.join(OUTPUT_BASE_DIR, level, "labels", "train"),
            },
            "test": {
                "img": os.path.join(OUTPUT_BASE_DIR, level, "images", "test"),
                "lbl": os.path.join(OUTPUT_BASE_DIR, level, "labels", "test"),
            },
            "val": {
                "img": os.path.join(OUTPUT_BASE_DIR, level, "images", "val"),
                "lbl": os.path.join(OUTPUT_BASE_DIR, level, "labels", "val"),
            }
        }

        ods_dirs = {
            "train": os.path.join(OUTPUT_BASE_DIR, level, "ods_values", "train"),
            "test": os.path.join(OUTPUT_BASE_DIR, level, "ods_values", "test"),
            "val": os.path.join(OUTPUT_BASE_DIR, level, "ods_values", "val")
        }

        for subset in ["train", "test", "val"]:
            os.makedirs(level_dirs[subset]["img"], exist_ok=True)
            os.makedirs(level_dirs[subset]["lbl"], exist_ok=True)
            os.makedirs(ods_dirs[subset], exist_ok=True)

        # 分层切分，避免 val/test 难度漂移太大
        train_split, test_split, val_split = stratified_split(
            samples_list,
            train_ratio=TRAIN_RATIO,
            test_ratio=TEST_RATIO,
            val_ratio=VAL_RATIO,
            num_bins=10
        )

        split_mapping = {
            "train": train_split,
            "test": test_split,
            "val": val_split
        }

        for subset, subset_samples in split_mapping.items():
            for idx, sample in enumerate(subset_samples):
                target_img_dir = level_dirs[subset]["img"]
                target_lbl_dir = level_dirs[subset]["lbl"]
                target_ods_dir = ods_dirs[subset]

                final_img_name = f"{level.lower()}_{subset}_{idx + 1:06d}.jpg"
                final_lbl_name = f"{level.lower()}_{subset}_{idx + 1:06d}.txt"

                final_img_path = os.path.join(target_img_dir, final_img_name)
                final_lbl_path = os.path.join(target_lbl_dir, final_lbl_name)

                os.replace(sample["image_path"], final_img_path)
                os.replace(sample["label_path"], final_lbl_path)

                ods_file_path = os.path.join(target_ods_dir, f"{os.path.splitext(final_img_name)[0]}.ods")
                with open(ods_file_path, "w", encoding="utf-8") as f:
                    f.write(f"{sample['ODS']:.6f}")

        print(f"  {level} 划分完成: train={len(train_split)}, test={len(test_split)}, val={len(val_split)}")

    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n正在为每个级别生成 YAML 配置文件...")
    for level in ["Hard", "Medium", "Easy"]:
        data_yaml_path = os.path.join(OUTPUT_BASE_DIR, level, "data.yaml")
        level_path = os.path.join(OUTPUT_BASE_DIR, level).replace("\\", "/")

        yaml_data = {
            "path": level_path,
            "train": f"{level_path}/images/train",
            "val": f"{level_path}/images/val",
            "test": f"{level_path}/images/test",
            "nc": 1,
            "names": ["mosquito"]
        }

        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"  {level} 的 data.yaml 已保存至: {data_yaml_path}")

    print(f"\n所有数据集生成完毕！")
    print(f"输出目录：{OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()