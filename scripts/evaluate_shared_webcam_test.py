from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate unified, digit-only, and letter-only YOLO models on "
            "a shared webcam test set using symbol-based class matching."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to shared test dataset data.yaml.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val", "train"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--unified-model",
        type=Path,
        required=True,
        help="Path to unified YOLO model weights.",
    )
    parser.add_argument(
        "--digits-model",
        type=Path,
        required=True,
        help="Path to digit-only YOLO model weights.",
    )
    parser.add_argument(
        "--letters-model",
        type=Path,
        required=True,
        help="Path to letter-only YOLO model weights.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence for accepting a detection.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold used during non-maximum suppression.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device for inference, for example 0 or cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("shared_webcam_test_results"),
        help="Directory for evaluation reports.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images for incorrect or missed detections.",
    )

    return parser.parse_args()


def load_dataset_config(
    data_yaml: Path,
) -> dict:
    with data_yaml.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if "names" not in config:
        raise ValueError(
            f"No class names found in {data_yaml}"
        )

    return config


def resolve_dataset_root(
    data_yaml: Path,
    config: dict,
) -> Path:
    dataset_root = Path(
        config.get("path", ".")
    )

    if not dataset_root.is_absolute():
        dataset_root = (
            data_yaml.parent / dataset_root
        )

    return dataset_root.resolve()


def get_image_label_pairs(
    data_yaml: Path,
    config: dict,
    split: str,
) -> list[tuple[Path, Path]]:
    dataset_root = resolve_dataset_root(
        data_yaml,
        config,
    )

    split_value = config.get(split)

    if split_value is None:
        raise ValueError(
            f"Split '{split}' is not defined in {data_yaml}"
        )

    split_path = dataset_root / split_value

    if split_path.is_file():
        raise NotImplementedError(
            "Split defined as a text file is not supported by this "
            "evaluation script. Use an images directory instead."
        )

    if not split_path.is_dir():
        raise FileNotFoundError(split_path)

    labels_root = dataset_root / (
        config.get(
            "labels",
            "labels",
        )
    )

    pairs = []

    for image_path in sorted(
        split_path.rglob("*")
    ):
        if not image_path.is_file():
            continue

        if (
            image_path.suffix.lower()
            not in IMAGE_SUFFIXES
        ):
            continue

        relative_image = image_path.relative_to(
            split_path
        )
        label_path = (
            labels_root
            / split
            / relative_image.parent
            / f"{image_path.stem}.txt"
        )

        pairs.append(
            (
                image_path,
                label_path,
            )
        )

    if not pairs:
        raise FileNotFoundError(
            f"No images found in {split_path}"
        )

    return pairs


def read_ground_truth(
    label_path: Path,
    class_names: list[str],
) -> list[dict]:
    ground_truth = []

    if not label_path.is_file():
        return ground_truth

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid YOLO label line in {label_path}: {line}"
            )

        class_id = int(values[0])

        if (
            class_id < 0
            or class_id >= len(class_names)
        ):
            raise ValueError(
                f"Class ID {class_id} is outside the shared dataset "
                f"class range in {label_path}"
            )

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        ground_truth.append(
            {
                "class_id": class_id,
                "symbol": class_names[class_id],
                "box": (
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                ),
            }
        )

    return ground_truth


def yolo_to_xyxy(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x_center, y_center, box_width, box_height = box

    x1 = (
        x_center - box_width / 2.0
    ) * width
    y1 = (
        y_center - box_height / 2.0
    ) * height
    x2 = (
        x_center + box_width / 2.0
    ) * width
    y2 = (
        y_center + box_height / 2.0
    ) * height

    return x1, y1, x2, y2


def calculate_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )
    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union_area = (
        area_a + area_b - intersection_area
    )

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def get_best_prediction(
    model,
    result,
    ground_truth_symbol: str,
    ground_truth_box: tuple[
        float, float, float, float
    ],
    width: int,
    height: int,
    iou_threshold: float,
) -> dict | None:
    """
    Return the highest-IoU model prediction whose symbol matches the
    ground-truth symbol.
    """
    if result.boxes is None:
        return None

    ground_truth_xyxy = yolo_to_xyxy(
        ground_truth_box,
        width,
        height,
    )

    best_prediction = None
    best_iou = 0.0

    for box_index in range(
        len(result.boxes.cls)
    ):
        class_id = int(
            result.boxes.cls[
                box_index
            ].item()
        )

        if class_id not in model.names:
            continue

        predicted_symbol = model.names[
            class_id
        ]

        if (
            predicted_symbol
            != ground_truth_symbol
        ):
            continue

        confidence = float(
            result.boxes.conf[
                box_index
            ].item()
        )

        if confidence <= 0.0:
            continue

        x1, y1, x2, y2 = (
            result.boxes.xyxy[
                box_index
            ].tolist()
        )

        iou = calculate_iou(
            ground_truth_xyxy,
            (
                x1,
                y1,
                x2,
                y2,
            ),
        )

        if (
            iou >= iou_threshold
            and iou > best_iou
        ):
            best_iou = iou
            best_prediction = {
                "class_id": class_id,
                "symbol": predicted_symbol,
                "confidence": confidence,
                "iou": iou,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            }

    return best_prediction


def evaluate_symbol_model(
    model_weights: Path,
    image_label_pairs: list[
        tuple[Path, Path]
    ],
    shared_class_names: list[str],
    args,
    run_name: str,
    allowed_symbols: set[str],
) -> dict:
    model = YOLO(str(model_weights))

    model_symbols = set(
        model.names.values()
    )
    evaluated_symbols = (
        allowed_symbols & model_symbols
    )

    if not evaluated_symbols:
        raise ValueError(
            f"No overlapping symbols between model {model_weights} and "
            f"allowed symbols {sorted(allowed_symbols)}"
        )

    per_symbol = {
        symbol: {
            "ground_truth": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "confidence_sum": 0.0,
            "iou_sum": 0.0,
        }
        for symbol in sorted(
            evaluated_symbols
        )
    }

    total_ground_truth = 0
    total_true_positives = 0
    total_false_negatives = 0
    total_images = 0
    total_images_with_ground_truth = 0

    visualization_root = (
        args.output / run_name / "visualizations"
    )

    if args.visualize:
        visualization_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    for image_path, label_path in image_label_pairs:
        ground_truth = read_ground_truth(
            label_path,
            shared_class_names,
        )
        ground_truth = [
            item
            for item in ground_truth
            if item["symbol"]
            in evaluated_symbols
        ]

        if not ground_truth:
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise OSError(
                f"Could not read image: {image_path}"
            )

        height, width = image.shape[:2]
        total_images += 1
        total_images_with_ground_truth += 1

        result = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        matched_ground_truth = set()
        visualization = (
            image.copy()
            if args.visualize
            else None
        )

        for object_index, object_data in enumerate(
            ground_truth
        ):
            symbol = object_data["symbol"]
            per_symbol[symbol][
                "ground_truth"
            ] += 1
            total_ground_truth += 1

            prediction = get_best_prediction(
                model,
                result,
                symbol,
                object_data["box"],
                width,
                height,
                args.iou,
            )

            if prediction is None:
                per_symbol[symbol][
                    "false_negatives"
                ] += 1
                total_false_negatives += 1

                if visualization is not None:
                    x1, y1, x2, y2 = (
                        yolo_to_xyxy(
                            object_data["box"],
                            width,
                            height,
                        )
                    )

                    cv2.rectangle(
                        visualization,
                        (
                            int(x1),
                            int(y1),
                        ),
                        (
                            int(x2),
                            int(y2),
                        ),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        visualization,
                        f"MISS {symbol}",
                        (
                            int(x1),
                            max(
                                20,
                                int(y1) - 8,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                continue

            per_symbol[symbol][
                "true_positives"
            ] += 1
            per_symbol[symbol][
                "confidence_sum"
            ] += prediction["confidence"]
            per_symbol[symbol][
                "iou_sum"
            ] += prediction["iou"]

            total_true_positives += 1
            matched_ground_truth.add(
                object_index
            )

            if visualization is not None:
                x1, y1, x2, y2 = prediction[
                    "box"
                ]

                cv2.rectangle(
                    visualization,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    visualization,
                    (
                        f"{symbol} "
                        f"{prediction['confidence']:.2f}"
                    ),
                    (
                        int(x1),
                        max(
                            20,
                            int(y1) - 8,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        if (
            visualization is not None
            and total_false_negatives > 0
        ):
            output_path = (
                visualization_root
                / image_path.name
            )
            cv2.imwrite(
                str(output_path),
                visualization,
            )

    recall = (
        total_true_positives
        / total_ground_truth
        if total_ground_truth
        else 0.0
    )

    per_symbol_metrics = {}

    for symbol, statistics in per_symbol.items():
        ground_truth_count = statistics[
            "ground_truth"
        ]
        true_positives = statistics[
            "true_positives"
        ]

        symbol_recall = (
            true_positives
            / ground_truth_count
            if ground_truth_count
            else 0.0
        )

        per_symbol_metrics[symbol] = {
            "ground_truth": (
                ground_truth_count
            ),
            "true_positives": (
                true_positives
            ),
            "false_negatives": statistics[
                "false_negatives"
            ],
            "recall": symbol_recall,
            "mean_confidence": (
                statistics[
                    "confidence_sum"
                ]
                / true_positives
                if true_positives
                else 0.0
            ),
            "mean_iou": (
                statistics["iou_sum"]
                / true_positives
                if true_positives
                else 0.0
            ),
        }

    results = {
        "model": str(model_weights),
        "evaluation_method": (
            "symbol_matching"
        ),
        "evaluated_symbols": sorted(
            evaluated_symbols
        ),
        "images_evaluated": total_images,
        "images_with_ground_truth": (
            total_images_with_ground_truth
        ),
        "ground_truth_objects": (
            total_ground_truth
        ),
        "true_positives": (
            total_true_positives
        ),
        "false_negatives": (
            total_false_negatives
        ),
        "recall": recall,
        "per_symbol": per_symbol_metrics,
    }

    report_directory = (
        args.output / run_name
    )
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_directory / "summary.json"
    )

    report_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n{run_name}")
    print(
        f"Model classes: {len(model.names)}"
    )
    print(
        f"Evaluated symbols: "
        f"{len(evaluated_symbols)}"
    )
    print(
        f"Ground-truth objects: "
        f"{total_ground_truth}"
    )
    print(
        f"True positives: {total_true_positives}"
    )
    print(
        f"False negatives: {total_false_negatives}"
    )
    print(f"Recall: {recall:.4f}")
    print(f"Report: {report_path}")

    return results


def main():
    args = parse_args()
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_dataset_config(
        args.data
    )

    names = config["names"]

    if isinstance(names, list):
        shared_class_names = [
            str(name) for name in names
        ]
    else:
        shared_class_names = [
            names[index]
            for index in sorted(names)
        ]

    image_label_pairs = (
        get_image_label_pairs(
            args.data,
            config,
            args.split,
        )
    )

    all_symbols = set(shared_class_names)
    digit_symbols = {
        symbol
        for symbol in all_symbols
        if symbol.isdigit()
    }
    letter_symbols = {
        symbol
        for symbol in all_symbols
        if symbol.isalpha()
    }

    results = {
        "dataset": str(args.data),
        "split": args.split,
        "shared_class_names": (
            shared_class_names
        ),
        "models": {},
    }

    results["models"][
        "unified"
    ] = evaluate_symbol_model(
        model_weights=args.unified_model,
        image_label_pairs=image_label_pairs,
        shared_class_names=shared_class_names,
        args=args,
        run_name="unified_model",
        allowed_symbols=all_symbols,
    )

    results["models"][
        "digits"
    ] = evaluate_symbol_model(
        model_weights=args.digits_model,
        image_label_pairs=image_label_pairs,
        shared_class_names=shared_class_names,
        args=args,
        run_name="digits_model",
        allowed_symbols=digit_symbols,
    )

    results["models"][
        "letters"
    ] = evaluate_symbol_model(
        model_weights=args.letters_model,
        image_label_pairs=image_label_pairs,
        shared_class_names=shared_class_names,
        args=args,
        run_name="letters_model",
        allowed_symbols=letter_symbols,
    )

    report_path = (
        args.output
        / "shared_webcam_test_summary.json"
    )

    report_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\nCombined report: {report_path}"
    )


if __name__ == "__main__":
    main()