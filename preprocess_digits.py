import os
import glob
import re
import cv2
import numpy as np

def preprocess_digit_image(
    image_path: str,
    target_size: tuple = (28, 28),
    inner_box_size: int = 20,
    use_center_of_mass: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust Multi-Stage Preprocessing Pipeline for Handwritten Digits (MNIST Standard):
    
    1. Grayscale Conversion & Illumination Normalization:
       - Multi-scale background division to remove shadows, non-uniform lighting, and paper color casts.
    2. Dual-Signal Ink Stroke Extraction:
       - Combines normalized division thresholding with morphological Black-Hat filtering
         to preserve fine ballpoint pen strokes while rejecting smooth shadow gradients.
    3. Morphological Enhancement:
       - Bridges broken pen stroke gaps and slightly thickens thin ink strokes.
    4. Smart Contour Filtering & Stroke Detection:
       - Rejects wide horizontal desk/page shadows and extreme corner margin artifacts.
       - Scores and selects the primary digit contour(s), applying proximity grouping for split strokes.
    5. Aspect-Ratio Preserving Rescaling:
       - Fits the cropped digit within a 20x20 bounding box without distortion.
    6. Canvas Placement & Centering:
       - Places onto a 28x28 black canvas.
       - Aligns via Center of Mass (Moments) matching official MNIST specifications.
    7. Normalization:
       - Returns (normalized_float32_array [0.0, 1.0], uint8_canvas [0, 255]).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # 1. Illumination Normalization via Background Division
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)

    # 2. Black-Hat Morphology to isolate dark pen strokes from background
    k_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    blackhat = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, k_stroke)

    # 3. Dual-Signal Ink Mask
    # Method A: Normalized ink + blackhat for faint pen strokes
    ink_mask_a = ((norm < 225) & (blackhat > 8)).astype(np.uint8) * 255
    # Method B: Direct contrast for high-contrast strokes
    ink_mask_b = ((norm < 210)).astype(np.uint8) * 255
    ink_combined = cv2.bitwise_or(ink_mask_a, ink_mask_b)

    # Bridge small stroke gaps & thicken slightly for standard MNIST stroke width
    ink_clean = cv2.morphologyEx(ink_combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    ink_enhanced = cv2.dilate(ink_clean, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # 4. Find Contours
    contours, _ = cv2.findContours(ink_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Score and filter contours to isolate the handwritten digit
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        # Reject full-span border lines or massive shadows
        if w > 0.45 * w_img and w > 2.2 * h:
            continue
        if h > 0.85 * h_img and h > 5.0 * w:
            continue
        if area < 25 and max(w, h) < 20:
            continue

        # Corner margin artifact rejection (e.g. tiny stray marks at extreme corners)
        in_corner = (x < 0.06 * w_img or x + w > 0.94 * w_img) and (y < 0.06 * h_img or y + h > 0.94 * h_img)
        if in_corner and max(w, h) < 0.12 * max(h_img, w_img):
            continue

        # Score formula:
        # 1. Base area
        # 2. General aspect ratio check (0.5 <= h/w <= 2.8 is typical for digits)
        # 3. Center proximity weighting
        aspect = h / max(1, w)
        dist_center = np.hypot((x + w / 2.0) - w_img / 2.0, (y + h / 2.0) - h_img / 2.0) / np.hypot(w_img / 2.0, h_img / 2.0)
        
        score = area / (1.0 + 1.5 * dist_center)
        if 0.5 <= aspect <= 2.8:
            score *= 1.5  # Normal handwritten digit aspect ratio
        elif aspect < 0.35:
            score *= 0.15  # Strongly penalize flat horizontal lines/shadows

        candidates.append((score, c, x, y, w, h, area))

    if not candidates:
        # Fallback: take largest contour if any
        if contours:
            candidates = [(cv2.contourArea(c), c, *cv2.boundingRect(c), cv2.contourArea(c)) for c in contours]
        else:
            canvas = np.zeros(target_size, dtype=np.uint8)
            return canvas.astype(np.float32) / 255.0, canvas

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_cnt = candidates[0][1]
    bx, by, bw, bh = candidates[0][2], candidates[0][3], candidates[0][4], candidates[0][5]

    # Proximity Grouping: group nearby split strokes or serifs that belong to the digit
    digit_cnts = [best_cnt]
    for item in candidates[1:]:
        _, c, cx, cy, cw, ch, _ = item
        dist_x = max(0, max(bx - (cx + cw), cx - (bx + bw)))
        dist_y = max(0, max(by - (cy + ch), cy - (by + bh)))
        max_dim = max(bw, bh)
        if dist_x < 0.25 * max_dim and dist_y < 0.25 * max_dim:
            digit_cnts.append(c)

    all_pts = np.vstack(digit_cnts)
    bx, by, bw, bh = cv2.boundingRect(all_pts)
    digit_crop = ink_enhanced[by : by + bh, bx : bx + bw]

    # 5. Aspect-Ratio Preserving Scaling (Fitting into inner 20x20 box)
    scale = float(inner_box_size) / max(bw, bh)
    nw = max(1, int(round(bw * scale)))
    nh = max(1, int(round(bh * scale)))

    resized = cv2.resize(
        digit_crop,
        (nw, nh),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    )

    # 6. Geometric Placement on 28x28 Black Canvas
    canvas = np.zeros(target_size, dtype=np.uint8)
    pt = (target_size[0] - nh) // 2
    pl = (target_size[1] - nw) // 2
    canvas[pt : pt + nh, pl : pl + nw] = resized

    # Center of Mass (Moments) alignment (official MNIST standard)
    if use_center_of_mass:
        m = cv2.moments(canvas)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            sx = int(round((target_size[1] / 2.0 - 0.5) - cx))
            sy = int(round((target_size[0] / 2.0 - 0.5) - cy))
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            canvas = cv2.warpAffine(canvas, M, target_size, flags=cv2.INTER_NEAREST, borderValue=0)

    # 7. Normalization to [0.0, 1.0] float32
    normalized_img = canvas.astype(np.float32) / 255.0
    return normalized_img, canvas


def process_dataset(
    input_dir: str,
    output_dir: str,
    digit_label: int = 0,
    exclude_copies: bool = True,
    save_png: bool = True,
    save_npy: bool = True,
    save_combined_dataset: bool = True,
    create_grid_preview: bool = True
):
    """
    Processes all digit images in input_dir, creates clean 28x28 preprocessed outputs,
    and packages them with labels into a single directory ready for ML training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect images
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

    image_paths = sorted(list(set(image_paths)))
    
    # Filter out output directory images and duplicate "- Copy" files
    filtered_paths = []
    for p in image_paths:
        abs_p = os.path.abspath(p)
        if os.path.abspath(output_dir) in abs_p:
            continue
        if exclude_copies and " - Copy" in os.path.basename(p):
            continue
        filtered_paths.append(p)

    image_paths = filtered_paths

    if not image_paths:
        print(f"[!] No valid input images found in {input_dir}")
        return None, None

    print(f"[*] Found {len(image_paths)} unique images to process in '{input_dir}' (digit label: {digit_label})...")
    print(f"[*] Saving preprocessed dataset to: {output_dir}")

    all_normalized_arrays = []
    display_canvases = []
    processed_filenames = []
    labels_list = []

    for idx, path in enumerate(image_paths, start=1):
        filename = os.path.basename(path)
        base_name, _ = os.path.splitext(filename)

        try:
            norm_arr, uint8_canvas = preprocess_digit_image(path)

            if save_png:
                png_path = os.path.join(output_dir, f"{base_name}_28x28.png")
                cv2.imwrite(png_path, uint8_canvas)

            if save_npy:
                npy_path = os.path.join(output_dir, f"{base_name}.npy")
                np.save(npy_path, norm_arr)

            all_normalized_arrays.append(norm_arr)
            display_canvases.append(uint8_canvas)
            processed_filenames.append(filename)
            labels_list.append(digit_label)

            nz = (uint8_canvas > 0).sum()
            print(f" [{idx:03d}/{len(image_paths):03d}] {filename} -> 28x28 (non-zero px: {nz})")
        except Exception as e:
            print(f" [!] Error processing {filename}: {e}")

    # Save per-digit combined .npz dataset
    if save_combined_dataset and all_normalized_arrays:
        dataset_stack = np.stack(all_normalized_arrays, axis=0)
        labels_arr = np.array(labels_list, dtype=np.int64)
        combined_path = os.path.join(output_dir, f"dataset_{digit_label}.npz")
        np.savez_compressed(
            combined_path,
            images=dataset_stack,
            labels=labels_arr,
            filenames=np.array(processed_filenames)
        )
        print(f"\n[+] Combined digit {digit_label} dataset saved: {combined_path}")
        print(f"    -> Images shape: {dataset_stack.shape} ({dataset_stack.dtype})")
        print(f"    -> Labels shape: {labels_arr.shape} (values: {np.unique(labels_arr)})")
        print(f"    -> Filenames count: {len(processed_filenames)}")

    # Create visual preview grid of all processed images (10x10 grid)
    if create_grid_preview and display_canvases:
        n_samples = len(display_canvases)
        grid_cols = 10
        grid_rows = (n_samples + grid_cols - 1) // grid_cols
        
        row_imgs = []
        cell_size = 84  # Upscaled for crisp viewing
        
        for r in range(grid_rows):
            col_imgs = []
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if idx < n_samples:
                    cell = cv2.resize(display_canvases[idx], (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
                    cv2.rectangle(cell, (0, 0), (cell_size - 1, cell_size - 1), (60, 60, 60), 1)
                else:
                    cell = np.zeros((cell_size, cell_size), dtype=np.uint8)
                col_imgs.append(cell)
            row_imgs.append(np.hstack(col_imgs))
            
        full_montage = np.vstack(row_imgs)
        montage_path = os.path.join(output_dir, f"preview_grid_{digit_label}.png")
        cv2.imwrite(montage_path, full_montage)
        print(f"[+] 10x10 Visual preview grid saved to: {montage_path}")

    print(f"\n[OK] Digit {digit_label} preprocessing completed successfully!\n")
    return all_normalized_arrays, labels_list


def process_all_available_digits(
    raw_base_dir: str = "./Digit_Raw",
    output_base_dir: str = "./Digit_Preprocessed"
):
    """
    Scans Digit_Raw/ for all digit folders (e.g. digit_0, digit_1, etc.),
    processes each digit into its corresponding Digit_Preprocessed/digit_X/ folder,
    and creates a unified master dataset_all_digits.npz in Digit_Preprocessed/.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    subdirs = sorted([d for d in os.listdir(raw_base_dir) if os.path.isdir(os.path.join(raw_base_dir, d))])
    
    all_images = []
    all_labels = []
    
    for sub in subdirs:
        # Infer digit label from folder name (e.g. 'digit_0' -> 0, 'digit_1' -> 1)
        match = re.search(r"\d+", sub)
        if match:
            label = int(match.group(0))
        else:
            continue
            
        in_dir = os.path.join(raw_base_dir, sub)
        out_dir = os.path.join(output_base_dir, sub)
        
        imgs, lbls = process_dataset(
            input_dir=in_dir,
            output_dir=out_dir,
            digit_label=label,
            exclude_copies=True,
            save_png=True,
            save_npy=True,
            save_combined_dataset=True,
            create_grid_preview=True
        )
        if imgs:
            all_images.extend(imgs)
            all_labels.extend(lbls)
            
    # If multiple digits exist, build master multi-class dataset
    if all_images:
        master_stack = np.stack(all_images, axis=0)
        master_labels = np.array(all_labels, dtype=np.int64)
        master_path = os.path.join(output_base_dir, "dataset_all_digits.npz")
        np.savez_compressed(
            master_path,
            images=master_stack,
            labels=master_labels
        )
        print(f"[+] Master multi-class dataset saved: {master_path}")
        print(f"    -> Total images: {master_stack.shape[0]}, classes present: {np.unique(master_labels)}")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_BASE = os.path.join(CURRENT_DIR, "Digit_Raw")
    PREP_BASE = os.path.join(CURRENT_DIR, "Digit_Preprocessed")

    # If subfolders like digit_0 exist in Digit_Raw, process them
    if os.path.exists(RAW_BASE):
        process_all_available_digits(RAW_BASE, PREP_BASE)
    else:
        # Fallback to single directory
        process_dataset(
            input_dir=CURRENT_DIR,
            output_dir=PREP_BASE,
            digit_label=0
        )
