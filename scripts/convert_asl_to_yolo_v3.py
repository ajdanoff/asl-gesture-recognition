# scripts/convert_asl_to_yolo_v3.py
import os
import cv2
import numpy as np
from pathlib import Path
import random

DATASET_ROOT = Path("datasets/asl-alphabet-dataset")  # adjust if needed
OUTPUT_ROOT = Path("datasets/asl_yolo")
VIS_ROOT = OUTPUT_ROOT / "vis"

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

random.seed(42)

def find_hand_box(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 50])
    upper = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 500:
        return None

    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def process_class_folder(folder_name, class_name, class_id, visualize=False):
    class_dir = DATASET_ROOT / folder_name
    if not class_dir.exists():
        print("Missing:", class_dir)
        return 0, 0

    images_out = OUTPUT_ROOT / "images"
    labels_out = OUTPUT_ROOT / "labels"
    vis_out = VIS_ROOT if visualize else None

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if vis_out:
        vis_out.mkdir(parents=True, exist_ok=True)

    img_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    random.shuffle(img_paths)

    n = len(img_paths)
    n_train = int(0.8 * n)
    train_paths = img_paths[:n_train]
    val_paths = img_paths[n_train:]

    train_count = 0
    val_count = 0

    def write_split(paths, split_name):
        nonlocal train_count, val_count
        split_images = images_out / split_name
        split_labels = labels_out / split_name
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        if visualize:
            vis_split = vis_out / split_name
            vis_split.mkdir(parents=True, exist_ok=True)

        for img_path in paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            box = find_hand_box(img)
            if box is None:
                cw, ch = int(w * 0.8), int(h * 0.8)
                x = (w - cw) // 2
                y = (h - ch) // 2
                box = (x, y, cw, ch)

            x, y, bw, bh = box

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            wn = bw / w
            hn = bh / h

            # Prefix filename with class letter to avoid collisions
            img_name = f"{class_name}_{img_path.stem}.jpg"

            cv2.imwrite(str(split_images / img_name), img)

            label_path = split_labels / (img_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            if visualize:
                vis_img = img.copy()
                cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    vis_img,
                    f"{class_name}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(str(vis_split / img_name), vis_img)

            if split_name == "train":
                train_count += 1
            else:
                val_count += 1

    write_split(train_paths, "train")
    write_split(val_paths, "val")

    return train_count, val_count

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        [d for d in DATASET_ROOT.iterdir() if d.is_dir() and d.name.endswith("-samples")]
    )
    expected_prefixes = [f"{c}-samples" for c in CLASS_NAMES]
    folders_sorted = []
    for prefix in expected_prefixes:
        matched = [f for f in folders if f.name == prefix]
        if matched:
            folders_sorted.append(matched[0])
        else:
            print("Warning: missing folder", prefix)

    total_train = 0
    total_val = 0

    for folder in folders_sorted:
        class_name = folder.name.split("-")[0]
        class_id = CLASS_TO_ID[class_name]
        print("Processing:", folder.name, "-> class", class_name, class_id)
        tr, va = process_class_folder(
            folder.name, class_name, class_id, visualize=True
        )
        total_train += tr
        total_val += va

    print("Total train images:", total_train)
    print("Total val images:", total_val)

    data_yaml = OUTPUT_ROOT / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write("train: ../datasets/asl_yolo/images/train\n")
        f.write("val: ../datasets/asl_yolo/images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names: [" + ", ".join(f"'{c}'" for c in CLASS_NAMES) + "]\n")
    print("Conversion done. data.yaml:", data_yaml)
    print("Visualizations saved to:", VIS_ROOT)

if __name__ == "__main__":
    main()