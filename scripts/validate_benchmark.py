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

TEST_DIR = "val"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--data-yaml", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    data_yaml = Path(args.data_yaml).resolve()

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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

    test_split = config.get(TEST_DIR)
    if not test_split:
        raise SystemExit("data.yaml does not define a test split")

    image_dir = (dataset_dir / test_split).resolve()
    if not image_dir.exists():
        raise SystemExit(f"Test image directory not found: {image_dir}")

    label_dir = dataset_dir / "labels" / TEST_DIR
    if not label_dir.exists():
        raise SystemExit(f"Test label directory not found: {label_dir}")

    images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise SystemExit("No benchmark images found")

    missing_labels = []
    invalid_class_ids = []
    total_instances = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        with label_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise SystemExit(
                        f"Invalid YOLO label line: {label_path}:{line_number}"
                    )

                class_id = int(fields[0])
                if class_id < 0 or class_id >= len(SHARED_CLASSES):
                    invalid_class_ids.append(
                        f"{label_path}:{line_number}: {class_id}"
                    )
                else:
                    total_instances += 1

    report = {
        "dataset_dir": str(dataset_dir),
        "image_count": len(images),
        "label_count": len(images) - len(missing_labels),
        "ground_truth_instances": total_instances,
        "missing_labels": missing_labels,
        "invalid_class_ids": invalid_class_ids,
        "class_count": len(SHARED_CLASSES),
    }

    print(json.dumps(report, indent=2))

    if missing_labels:
        raise SystemExit(f"Missing labels for {len(missing_labels)} images")
    if invalid_class_ids:
        raise SystemExit(f"Invalid class IDs: {invalid_class_ids}")


if __name__ == "__main__":
    main()