from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import mediapipe as mp

CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-1", type=Path, default=Path("asl-alphabet-dataset"))
    parser.add_argument("--source-2", type=Path, default=Path("asl_unified_dataset"))
    parser.add_argument("--webcam-root", type=Path, default=Path("webcam_data"))
    parser.add_argument("--output", type=Path, default=Path("datasets/asl_unified_webcam_mp"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def get_source_1_samples(root: Path):
    for class_name in CLASS_NAMES:
        if not class_name.isalpha():
            continue
        folder = root / f"{class_name}-samples"
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield "source1", class_name, image_path


def get_source_2_samples(root: Path):
    for class_name in CLASS_NAMES:
        folder_name = class_name.lower()
        folder = root / folder_name
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield "source2", class_name, image_path


def get_webcam_gesture_samples(root: Path):
    # root/webcam_gestures/<CLASS>/
    gestures_root = root / "webcam_gestures"
    if not gestures_root.is_dir():
        return
    for class_name in CLASS_NAMES:
        folder = gestures_root / class_name
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield "webcam", class_name, image_path


def get_webcam_background_images(root: Path):
    # root/webcam_bg/*.jpg, *.png, ...
    bg_root = root / "webcam_bg"
    if not bg_root.is_dir():
        return
    for image_path in sorted(bg_root.iterdir()):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            yield "webcam_bg", None, image_path


def get_all_samples(source_1: Path, source_2: Path, webcam_root: Path):
    yield from get_source_1_samples(source_1)
    yield from get_source_2_samples(source_2)
    yield from get_webcam_gesture_samples(webcam_root)
    # Background images are handled separately (no labels)


def find_hand_box(image_bgr, landmarker):
    # Reuse your existing MediaPipe box function here.
    # Returns (x1, y1, x2, y2) or None.
    height, width = image_bgr.shape[:2]
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )
    
    result = landmarker.detect(mp_image)
    
    if not result.hand_landmarks:
        return None
    
    # Use the first detected hand.
    landmarks = result.hand_landmarks[0]
    
    x_values = [landmark.x * width for landmark in landmarks]
    y_values = [landmark.y * height for landmark in landmarks]
    
    x_min = max(0.0, min(x_values))
    y_min = max(0.0, min(y_values))
    x_max = min(float(width - 1), max(x_values))
    y_max = min(float(height - 1), max(y_values))
    
    box_width = max(1.0, x_max - x_min)
    box_height = max(1.0, y_max - y_min)
    
    # Include wrist and a small context margin.
    margin_x = 0.12 * box_width
    margin_y = 0.12 * box_height
    
    x1 = max(0, int(x_min - margin_x))
    y1 = max(0, int(y_min - margin_y))
    x2 = min(width - 1, int(x_max + margin_x))
    y2 = min(height - 1, int(y_max + margin_y))
    
    return x1, y1, x2, y2

def fallback_box(width: int, height: int):
    return 0, 0, width - 1, height - 1


def to_yolo_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = (x1 + x2) / 2.0 / width
    yc = (y1 + y2) / 2.0 / height
    return xc, yc, bw / width, bh / height


def output_name(source_name: str, class_name: str | None, image_path: Path, index: int):
    if class_name is None:
        return f"{source_name}_bg_{index:06d}.jpg"
    return f"{source_name}_{class_name}_{image_path.stem}.jpg"


def write_yaml(output_root: Path):
    yaml_path = output_root / "data.yaml"
    names = ", ".join(f"'{name}'" for name in CLASS_NAMES)
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(CLASS_NAMES)}",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    for p in [args.source_1, args.source_2, args.webcam_root]:
        if not p.is_dir():
            raise FileNotFoundError(p)

    if args.output.exists():
        shutil.rmtree(args.output)

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    if args.visualize:
        for split in ("train", "val"):
            (args.output / "vis" / split).mkdir(parents=True)

    # Load MediaPipe landmarker (reuse your existing code)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path="models/hand_landmarker.task",
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        # Collect labeled samples
        labeled_samples = list(get_all_samples(args.source_1, args.source_2, args.webcam_root))
        random.shuffle(labeled_samples)

        # Collect background images
        bg_images = list(get_webcam_background_images(args.webcam_root))
        random.shuffle(bg_images)

        counts = {"train": 0, "val": 0}
        fallback_count = 0

        # Process labeled images
        for index, (source_name, class_name, image_path) in enumerate(labeled_samples):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            h, w = image.shape[:2]
            box = find_hand_box(image, landmarker)
            if box is None:
                box = fallback_box(w, h)
                fallback_count += 1

            split = "val" if random.random() < args.val_ratio else "train"
            filename = output_name(source_name, class_name, image_path, index)

            out_img = args.output / "images" / split / filename
            out_lbl = args.output / "labels" / split / f"{Path(filename).stem}.txt"

            cv2.imwrite(str(out_img), image)

            xc, yc, bw, bh = to_yolo_box(box, w, h)
            cid = CLASS_TO_ID[class_name]
            out_lbl.write_text(
                f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n",
                encoding="utf-8",
            )

            if args.visualize:
                vis = image.copy()
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis, class_name, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                out_vis = args.output / "vis" / split / filename
                cv2.imwrite(str(out_vis), vis)

            counts[split] += 1

        # Process background images (no labels)
        for idx, (source_name, _, image_path) in enumerate(bg_images):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            split = "val" if random.random() < args.val_ratio else "train"
            filename = output_name(source_name, None, image_path, idx)
            out_img = args.output / "images" / split / filename
            # No label file written → background sample

            if args.visualize:
                vis = image.copy()
                cv2.putText(
                    vis, "BG", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                out_vis = args.output / "vis" / split / filename
                cv2.imwrite(str(out_vis), vis)

            counts[split] += 1

    write_yaml(args.output)

    print(f"Total samples: {sum(counts.values())}")
    print(f"Train: {counts['train']}")
    print(f"Val: {counts['val']}")
    print(f"MediaPipe fallbacks (labeled): {fallback_count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    #import pdb
    #pdb.set_trace()
    main()