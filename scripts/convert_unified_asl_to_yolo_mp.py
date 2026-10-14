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

CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-1",
        type=Path,
        default=Path("asl-alphabet-dataset"),
        help="First dataset containing A-samples, B-samples, etc.",
    )
    parser.add_argument(
        "--source-2",
        type=Path,
        default=Path("asl_unified_dataset"),
        help="Second dataset containing 0-9 and a-z.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/asl_unified_yolo_mp"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write images with bounding boxes under output/vis.",
    )
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


def get_samples(source_1: Path, source_2: Path):
    yield from get_source_1_samples(source_1)
    yield from get_source_2_samples(source_2)


def find_hand_box(image_bgr: cv2.Mat, landmarker):
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
    # The second dataset is already cropped, so this is a sensible fallback.
    return 0, 0, width - 1, height - 1


def to_yolo_box(box, width: int, height: int):
    x1, y1, x2, y2 = box

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    normalized_width = box_width / width
    normalized_height = box_height / height

    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, normalized_width)),
        max(0.0, min(1.0, normalized_height)),
    )


def output_name(source_name: str, class_name: str, image_path: Path):
    # Include source and class because original filenames may collide.
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

    if not args.source_1.is_dir():
        raise FileNotFoundError(args.source_1)

    if not args.source_2.is_dir():
        raise FileNotFoundError(args.source_2)

    if args.output.exists():
        shutil.rmtree(args.output)

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    if args.visualize:
        for split in ("train", "val"):
            (args.output / "vis" / split).mkdir(parents=True)

    samples = list(get_samples(args.source_1, args.source_2))
    random.shuffle(samples)

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
        counts = {"train": 0, "val": 0}
        fallback_count = 0

        for index, (source_name, class_name, image_path) in enumerate(samples):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: unable to read {image_path}")
                continue

            height, width = image.shape[:2]

            box = find_hand_box(image, landmarker)
            if box is None:
                box = fallback_box(width, height)
                fallback_count += 1

            split = "val" if random.random() < args.val_ratio else "train"
            filename = output_name(source_name, class_name, image_path)

            output_image = args.output / "images" / split / filename
            output_label = args.output / "labels" / split / f"{Path(filename).stem}.txt"

            # Re-encode all images as JPEG for consistent YOLO input.
            if not cv2.imwrite(str(output_image), image):
                raise RuntimeError(f"Unable to write {output_image}")

            x_center, y_center, box_width, box_height = to_yolo_box(
                box,
                width,
                height,
            )

            class_id = CLASS_TO_ID[class_name]
            output_label.write_text(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n",
                encoding="utf-8",
            )

            if args.visualize:
                visualization = image.copy()
                x1, y1, x2, y2 = box

                cv2.rectangle(
                    visualization,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    visualization,
                    class_name,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                output_vis = args.output / "vis" / split / filename
                cv2.imwrite(str(output_vis), visualization)

            counts[split] += 1

            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1}/{len(samples)}")

    write_yaml(args.output)

    print(f"Total samples: {sum(counts.values())}")
    print(f"Train: {counts['train']}")
    print(f"Val: {counts['val']}")
    print(f"MediaPipe fallbacks: {fallback_count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()