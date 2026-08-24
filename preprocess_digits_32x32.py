"""
Standardized 32x32 Handwritten Digit Dataset Pipeline & Augmentation
====================================================================
Pipeline Steps:
  1. RAW PHOTOS (from Digit_Raw/digit_X/)
  2. Quality control (ink stroke vs shadow check)
  3. Crop digit
  4. Grayscale
  5. Lighting / background division normalization
  6. Center digit via Moments (Center of Mass)
  7. Resize to 32x32 with aspect ratio preservation
  8. Normalize to float32 [0.0, 1.0]
  9. Stratified Train (70%) / Validation (15%) / Test (15%) split
 10. Data Augmentation ONLY on training split
"""

import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_digit_32x32(image_path, target_size=(32, 32), inner_box_size=24):
    """
    Standardizes raw digit image into 32x32 MNIST-like canvas.
    Digit fits inside 24x24 box and is centered via Center of Mass.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 2. Lighting / Background Division
    k_size = max(31, (min(h, w) // 10) | 1)
    bg = cv2.GaussianBlur(gray, (k_size, k_size), 0)
    norm = cv2.divide(gray, bg, scale=255)

    # 3. Dual-signal ink extraction
    k_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, k_stroke)
    ink_mask_a = ((norm < 225) & (blackhat > 8)).astype(np.uint8) * 255
    ink_mask_b = ((norm < 210)).astype(np.uint8) * 255
    ink = cv2.bitwise_or(ink_mask_a, ink_mask_b)

    # Bridge stroke gaps
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_bridge)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # 4. Find digit contours
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(target_size, dtype=np.float32), np.zeros(target_size, dtype=np.uint8)

    valid_cnts = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if bw > 2.8 * bh:  # horizontal line / desk shadow
            continue
        if bh > 4.5 * bw:  # vertical artifact
            continue
        if area < 15 and max(bw, bh) < 15:  # speck noise
            continue
        valid_cnts.append((c, area, bx, by, bw, bh))

    if not valid_cnts:
        valid_cnts = [(c, cv2.contourArea(c), *cv2.boundingRect(c)) for c in cnts]

    valid_cnts.sort(key=lambda item: item[1], reverse=True)

    # Proximity merge
    primary = valid_cnts[0]
    _, _, p_x, p_y, p_w, p_h = primary
    min_x, min_y = p_x, p_y
    max_x, max_y = p_x + p_w, p_y + p_h

    margin = int(max(p_w, p_h) * 0.45)
    for c, area, bx, by, bw, bh in valid_cnts[1:]:
        if (bx + bw >= p_x - margin and bx <= p_x + p_w + margin and
            by + bh >= p_y - margin and by <= p_y + p_h + margin):
            min_x = min(min_x, bx)
            min_y = min(min_y, by)
            max_x = max(max_x, bx + bw)
            max_y = max(max_y, by + bh)

    crop = ink[min_y:max_y, min_x:max_x]
    ch, cw = crop.shape
    if ch == 0 or cw == 0:
        return np.zeros(target_size, dtype=np.float32), np.zeros(target_size, dtype=np.uint8)

    # 5. Aspect-ratio preserving scale to inner_box_size (e.g. 24x24 inside 32x32)
    scale = inner_box_size / max(ch, cw)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # Place in 32x32 canvas
    canvas = np.zeros(target_size, dtype=np.uint8)
    sy = (target_size[0] - nh) // 2
    sx = (target_size[1] - nw) // 2
    canvas[sy:sy+nh, sx:sx+nw] = resized

    # 6. Center via Moments (Center of Mass)
    M = cv2.moments(canvas)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        target_cx, target_cy = target_size[1] / 2.0, target_size[0] / 2.0
        shift_x = int(round(target_cx - cx))
        shift_y = int(round(target_cy - cy))
        shift_x = max(-3, min(3, shift_x))
        shift_y = max(-3, min(3, shift_y))
        T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, T, target_size, flags=cv2.INTER_NEAREST)

    # 7. Normalize [0.0, 1.0]
    norm_arr = canvas.astype(np.float32) / 255.0
    return norm_arr, canvas


def augment_image_32x32(img_28):
    """
    Generates synthetic variants with small rotations, translations, and scaling.
    img_28: uint8 array (32, 32)
    """
    augmented = []
    h, w = img_28.shape

    # 1. Rotations (-12, -6, +6, +12 deg)
    for angle in [-12, -6, 6, 12]:
        M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(img_28, M_rot, (w, h), flags=cv2.INTER_LINEAR)
        augmented.append(rot)

    # 2. Small translations (+-2 px)
    for tx, ty in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        trans = cv2.warpAffine(img_28, M_trans, (w, h), flags=cv2.INTER_NEAREST)
        augmented.append(trans)

    # 3. Slight dilation / erosion (stroke thickness variations)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dil = cv2.dilate(img_28, k, iterations=1)
    augmented.append(dil)

    return augmented


def build_complete_32x32_dataset(raw_base_dir="Digit_Raw", output_base_dir="Digit_Dataset_32x32"):
    """
    Compiles complete 32x32 dataset with Stratified 70/15/15 split and Train-only Augmentation.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    all_images_dir = os.path.join(output_base_dir, "all_images")
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "val")
    test_dir = os.path.join(output_base_dir, "test")

    for d in [all_images_dir, train_dir, val_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        for digit in range(10):
            os.makedirs(os.path.join(d, str(digit)), exist_ok=True)

    # Collect and preprocess all raw digits
    samples = []
    print("=== Step 1: Preprocessing All Raw Images to 32x32 Standardized Canvas ===")
    for digit in range(10):
        digit_raw_folder = os.path.join(raw_base_dir, f"digit_{digit}")
        if not os.path.exists(digit_raw_folder):
            continue

        raw_files = sorted(glob.glob(os.path.join(digit_raw_folder, "*.*")))
        print(f"  Digit {digit}: Found {len(raw_files)} raw files in {digit_raw_folder}")

        for i, rpath in enumerate(raw_files, start=1):
            norm_arr, uint8_canvas = preprocess_digit_32x32(rpath)
            clean_id = f"digit_{digit}_{i:04d}.png"
            # Save in all_images
            cv2.imwrite(os.path.join(all_images_dir, str(digit), clean_id), uint8_canvas)

            samples.append({
                "image_id": clean_id,
                "digit_label": digit,
                "source_file": os.path.basename(rpath),
                "canvas_uint8": uint8_canvas,
                "norm_float32": norm_arr
            })

    df = pd.DataFrame(samples)
    print(f"\n[+] Total cleanly preprocessed samples: {len(df)}")

    # === Step 2: Stratified Train (70%), Val (15%), Test (15%) Split ===
    print("\n=== Step 2: Stratified Train (70%) / Val (15%) / Test (15%) Split ===")
    labels = df["digit_label"].values

    # Split 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=42, stratify=labels
    )
    temp_labels = labels[temp_idx]
    # Split 30% temp into 15% val, 15% test (50/50 of temp)
    val_idx_sub, test_idx_sub = train_test_split(
        np.arange(len(temp_idx)), test_size=0.50, random_state=42, stratify=temp_labels
    )
    val_idx = temp_idx[val_idx_sub]
    test_idx = temp_idx[test_idx_sub]

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Save original images into respective split folders
    for idx, row in df.iterrows():
        lbl = row["digit_label"]
        split = row["split"]
        fname = row["image_id"]
        img = row["canvas_uint8"]
        cv2.imwrite(os.path.join(output_base_dir, split, str(lbl), fname), img)

    # === Step 3: Data Augmentation ONLY on Training Set ===
    print("\n=== Step 3: Data Augmentation (Applied ONLY on Training Split) ===")
    aug_records = []
    train_df = df[df["split"] == "train"]

    for idx, row in train_df.iterrows():
        lbl = row["digit_label"]
        base_id = os.path.splitext(row["image_id"])[0]
        orig_img = row["canvas_uint8"]

        aug_variants = augment_image_32x32(orig_img)
        for v_idx, aug_img in enumerate(aug_variants, start=1):
            aug_id = f"{base_id}_aug_{v_idx:02d}.png"
            cv2.imwrite(os.path.join(train_dir, str(lbl), aug_id), aug_img)
            aug_records.append({
                "image_id": aug_id,
                "digit_label": lbl,
                "source_file": f"augmented_from_{row['image_id']}",
                "canvas_uint8": aug_img,
                "norm_float32": aug_img.astype(np.float32) / 255.0,
                "split": "train"
            })

    aug_df = pd.DataFrame(aug_records)
    print(f"  Generated {len(aug_df)} synthetic augmented samples for training.")

    # Combine training with augmented samples
    full_train_df = pd.concat([train_df, aug_df], ignore_index=True)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    # NumPy arrays
    X_train = np.stack(full_train_df["norm_float32"].values).astype(np.float32)
    y_train = full_train_df["digit_label"].values.astype(np.int64)

    X_val = np.stack(val_df["norm_float32"].values).astype(np.float32)
    y_val = val_df["digit_label"].values.astype(np.int64)

    X_test = np.stack(test_df["norm_float32"].values).astype(np.float32)
    y_test = test_df["digit_label"].values.astype(np.int64)

    # Save NPZ bundle
    npz_path = os.path.join(output_base_dir, "dataset_32x32.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_orig_all=np.stack(df["norm_float32"].values).astype(np.float32),
        y_orig_all=df["digit_label"].values.astype(np.int64)
    )

    # Metadata CSV
    all_metadata_df = pd.concat([df[["image_id", "digit_label", "split", "source_file"]],
                                 aug_df[["image_id", "digit_label", "split", "source_file"]]],
                                ignore_index=True)
    csv_path = os.path.join(output_base_dir, "dataset_metadata.csv")
    all_metadata_df.to_csv(csv_path, index=False)

    print("\n[+] Dataset Build Summary (32x32 Standard):")
    print(f"  -> Total Original Samples: {len(df)}")
    print(f"  -> Training Split (with Augmentations): {len(X_train)} samples ({X_train.shape})")
    print(f"  -> Validation Split (Untouched):        {len(X_val)} samples ({X_val.shape})")
    print(f"  -> Testing Split (Untouched Holdout):   {len(X_test)} samples ({X_test.shape})")
    print(f"  -> Saved NPZ Bundle: {npz_path}")
    print(f"  -> Saved Metadata CSV: {csv_path}")


if __name__ == "__main__":
    build_complete_32x32_dataset()
