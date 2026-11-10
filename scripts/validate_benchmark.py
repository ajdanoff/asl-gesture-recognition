#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


SHARED_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]


def main():
    parser = argparse.ArgumentParser(
        description="Validate an ASL webcam benchmark dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the downloaded benchmark dataset directory.",
    )
    parser.add_argument(
        "--data-yaml",
        required=True,
        help="Path to the benchmark data.yaml file.",
    )
    parser.add_argument(
        "--allow-background",
        action="store_true",
        default=True,
        help="Allow images without label files as background/no-gesture images.",
    )
    parser.add_argument(
        "--expected-gesture-images",
        type=int,
        default=None,
        help=(
            "Optional expected number of labeled gesture images. "
            "For the original benchmark, use 36."
        ),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    data_yaml = Path(args.data_yaml).resolve()

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names = config.get("names")

    if not names:
        raise SystemExit("data.yaml does not define class names")

    if isinstance(names, list):
        actual_names = [str(name) for name in names]
    else:
        actual_names = [str(names[index]) for index in sorted(names)]

    if actual_names != SHARED_CLASSES:
        raise SystemExit(
            "Class mapping mismatch.\n"
            f"Expected: {SHARED_CLASSES}\n"
            f"Found: {actual_names}"
        )

    test_split = config.get("val")

    if not test_split:
        raise SystemExit("data.yaml does not define a 'val' split")

    image_dir = (dataset_dir / test_split).resolve()

    if not image_dir.exists():
        raise SystemExit(f"Test image directory not found: {image_dir}")

    label_dir = dataset_dir / "labels" / "val"

    if not label_dir.exists():
        raise SystemExit(f"Test label directory not found: {label_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise SystemExit(f"No benchmark images found in: {image_dir}")

    labeled_images = []
    background_images = []
    invalid_label_files = []
    invalid_class_ids = []
    ground_truth_instances = 0

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            if args.allow_background:
                background_images.append(image_path.name)
                continue

            invalid_label_files.append(image_path.name)
            continue

        file_has_valid_annotation = False

        with label_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                fields = line.split()

                if len(fields) != 5:
                    invalid_label_files.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                try:
                    class_id = int(float(fields[0]))
                except ValueError:
                    invalid_label_files.append(
                        f"{label_path.name}:{line_number}"
                    )
                    continue

                if class_id < 0 or class_id >= len(SHARED_CLASSES):
                    invalid_class_ids.append(
                        f"{label_path.name}:{line_number}: {class_id}"
                    )
                    continue

                ground_truth_instances += 1
                file_has_valid_annotation = True

        if file_has_valid_annotation:
            labeled_images.append(image_path.name)
        else:
            background_images.append(image_path.name)

    report = {
        "dataset_dir": str(dataset_dir),
        "image_count": len(image_paths),
        "labeled_gesture_images": len(labeled_images),
        "background_images": len(background_images),
        "ground_truth_instances": ground_truth_instances,
        "class_count": len(SHARED_CLASSES),
        "invalid_label_files": invalid_label_files,
        "invalid_class_ids": invalid_class_ids,
        "background_image_names": background_images,
    }

    print(json.dumps(report, indent=2))

    if invalid_label_files:
        raise SystemExit(
            f"Invalid label files or lines: {invalid_label_files}"
        )

    if invalid_class_ids:
        raise SystemExit(f"Invalid class IDs: {invalid_class_ids}")

    if args.expected_gesture_images is not None:
        if len(labeled_images) != args.expected_gesture_images:
            raise SystemExit(
                f"Expected {args.expected_gesture_images} labeled gesture images, "
                f"but found {len(labeled_images)}."
            )

        if ground_truth_instances != args.expected_gesture_images:
            raise SystemExit(
                f"Expected {args.expected_gesture_images} ground-truth instances, "
                f"but found {ground_truth_instances}."
            )

    print(
        f"Validation passed: {len(labeled_images)} labeled gesture images, "
        f"{len(background_images)} background images, "
        f"{ground_truth_instances} ground-truth instances."
    )


if __name__ == "__main__":
    main()