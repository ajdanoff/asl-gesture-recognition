#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import yaml
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


SHARED_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]


MODEL_DOMAINS = {
    "unified": set(SHARED_CLASSES),
    "digits": {str(index) for index in range(10)},
    "letters": {SHARED_CLASSES[index] for index in range(10, 36)},
}


EXPECTED_GROUND_TRUTH = {
    "original": {
        "unified": 36,
        "digits": 10,
        "letters": 26,
    },
    "augmented": {
        "unified": 216,
        "digits": 60,
        "letters": 156,
    },
}


TEST_DIR = "val"


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def yolo_box_to_pixels(values, width, height):
    class_id, x_center, y_center, box_width, box_height = values

    x1 = (x_center - box_width / 2) * width
    y1 = (y_center - box_height / 2) * height
    x2 = (x_center + box_width / 2) * width
    y2 = (y_center + box_height / 2) * height

    return int(class_id), [x1, y1, x2, y2]

def resolve_model_path(model_id, model_filename):
    if model_filename and (model_filename.endswith(".pt") or model_filename.endswith(".pth")):
        candidate = Path(model_filename)

        if candidate.exists():
            return str(candidate.resolve())

        raise SystemExit(f"Local model file does not exist: {candidate}")

    repository_name = model_id.split("/")[-1]
    filename = model_filename or f"{repository_name.replace('-', '_')}_best.pt"

    print(f"Downloading model: {model_id}")
    print(f"Requested filename: {filename}")

    try:
        return hf_hub_download(
            repo_id=model_id,
            filename=filename,
        )
    except Exception as error:
        raise SystemExit(
            f"Could not download '{filename}' from Hugging Face repository "
            f"'{model_id}'.\n"
            "Check the exact filename in the repository's Files and versions tab, "
            "then pass it with --model-filename.\n"
            f"Original error: {error}"
        )

def main():
    import pdb
    # pdb.set_trace()
    
    parser = argparse.ArgumentParser(
        description="Evaluate an ASL YOLO detector on a shared webcam benchmark."
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
        "--model",
        required=True,
        choices=["unified", "digits", "letters"],
        help="Evaluation domain to use.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model repository ID, for example ajdanoff/asl-unified-digits-yolo11n-mp.",
    )
    parser.add_argument(
        "--model-filename",
        default=None,
        help=(
            "Filename inside the Hugging Face repository. "
            "If omitted, the script derives <repo_name_with_underscores>_best.pt."
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=["original", "augmented"],
        default=None,
        help=(
            "Benchmark type. If omitted, it is inferred from the dataset directory name. "
            "Used only for the optional ground-truth count check."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="Minimum IoU required for a symbol-matched true positive.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the JSON result file.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional path for a Markdown result summary.",
    )
    parser.add_argument(
        "--skip-ground-truth-check",
        action="store_true",
        help="Disable the expected ground-truth count check.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    data_yaml = Path(args.data_yaml).resolve()

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    if not data_yaml.exists():
        raise SystemExit(f"data.yaml does not exist: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if TEST_DIR not in config:
        raise SystemExit(f"data.yaml does not define '{TEST_DIR}'")

    test_split = config[TEST_DIR]
    image_dir = (dataset_dir / test_split).resolve()
    label_dir = (dataset_dir / "labels" / TEST_DIR).resolve()

    if not image_dir.exists():
        raise SystemExit(f"Image directory does not exist: {image_dir}")

    if not label_dir.exists():
        raise SystemExit(f"Label directory does not exist: {label_dir}")

    model_path = resolve_model_path(args.model_id, args.model_filename)

    model = YOLO(model_path)
    model_names = model.names

    valid_symbols = MODEL_DOMAINS[args.model]

    if args.benchmark is None:
        dataset_name = dataset_dir.name.lower()
        if "original" in dataset_name:
            benchmark = "original"
        elif "augmented" in dataset_name:
            benchmark = "augmented"
        else:
            benchmark = None
    else:
        benchmark = args.benchmark

    expected_ground_truth = None
    if benchmark and not args.skip_ground_truth_check:
        expected_ground_truth = EXPECTED_GROUND_TRUTH[benchmark][args.model]

    results = {
        "model": args.model,
        "model_id": args.model_id,
        "model_path": model_path,
        "dataset": dataset_dir.name,
        "benchmark": benchmark,
        "evaluation_domain": sorted(valid_symbols),
        "iou_threshold": args.iou_threshold,
        "confidence": args.confidence,
        "ground_truth": 0,
        "out_of_domain_ground_truth": 0,
        "true_positives": 0,
        "false_positives": 0,
        "background_images": 0,
        "background_false_positives": 0,
        "per_class": {},
        "missed_classes": [],
    }

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise SystemExit(f"No benchmark images found in: {image_dir}")

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        is_background = not label_path.exists()

        if is_background:
            label_path = None
            results["background_images"] += 1

        image = cv2.imread(str(image_path))

        if image is None:
            raise SystemExit(f"Could not read image: {image_path}")

        height, width = image.shape[:2]

        ground_truth = []

        if label_path is not None:
            with label_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue

                    fields = line.split()

                    if len(fields) != 5:
                        raise SystemExit(
                            f"Invalid YOLO label line: {label_path}:{line_number}"
                        )

                    class_id, box = yolo_box_to_pixels(
                        [float(value) for value in fields],
                        width,
                        height,
                    )

                    if class_id < 0 or class_id >= len(SHARED_CLASSES):
                        raise SystemExit(
                            f"Invalid class ID {class_id} in {label_path}:{line_number}"
                        )

                    symbol = SHARED_CLASSES[class_id]

                    if symbol not in valid_symbols:
                        results["out_of_domain_ground_truth"] += 1
                        continue

                    ground_truth.append((class_id, box, symbol))

        predictions = model.predict(
            source=str(image_path),
            conf=args.confidence,
            verbose=False,
        )[0]

        predicted = []

        if predictions.boxes is not None:
            for class_id, confidence, xyxy in zip(
                predictions.boxes.cls.tolist(),
                predictions.boxes.conf.tolist(),
                predictions.boxes.xyxy.tolist(),
            ):
                predicted.append(
                {
                    "class_id": int(class_id),
                    "symbol": str(model_names[int(class_id)]),
                    "confidence": float(confidence),
                    "box_xyxy": [float(value) for value in xyxy],
                    "matched": False,
                }
            )

        results["ground_truth"] += len(ground_truth)
        
        matched_ground_truth = set()
        matched_predictions = set()

        for gt_index, (gt_class_id, gt_box, gt_symbol) in enumerate(ground_truth):
            best_prediction = None
            best_iou = 0.0

            for pred_index, prediction in enumerate(predicted):
                if pred_index in matched_predictions:
                    continue

                pred_symbol = prediction["symbol"]
                pred_box = prediction["box_xyxy"]

                if pred_symbol != gt_symbol:
                    continue

                score = iou(gt_box, pred_box)

                if score >= args.iou_threshold and score > best_iou:
                    best_prediction = pred_index
                    best_iou = score

            if best_prediction is not None:
                matched_ground_truth.add(gt_index)
                matched_predictions.add(best_prediction)
                results["true_positives"] += 1

        for prediction_index in matched_predictions:
            predicted[prediction_index]["matched"] = True

        unmatched = [
            prediction
            for prediction in predicted
            if not prediction["matched"]
        ]

        if unmatched:
            results.setdefault("false_positive_details", []).append(
                {
                    "image": image_path.name,
                    "predictions": unmatched,
                }
            )

        results["false_positives"] += len(unmatched)

        if is_background:
            results["background_false_positives"] += len(unmatched)

        for gt_index, (_, _, symbol) in enumerate(ground_truth):
            stats = results["per_class"].setdefault(
                symbol,
                {
                    "ground_truth": 0,
                    "true_positives": 0,
                },
            )

            stats["ground_truth"] += 1

            if gt_index in matched_ground_truth:
                stats["true_positives"] += 1
                
    if (
        expected_ground_truth is not None
        and results["ground_truth"] != expected_ground_truth
    ):
        raise SystemExit(
            f"Ground-truth count mismatch for {args.model}: "
            f"expected {expected_ground_truth}, got {results['ground_truth']}. "
            "Check the benchmark dataset, data.yaml paths, and evaluation domain."
        )

    for symbol, stats in results["per_class"].items():
        stats["recall"] = (
            stats["true_positives"] / stats["ground_truth"]
            if stats["ground_truth"]
            else 0.0
        )

        if stats["true_positives"] == 0:
            results["missed_classes"].append(symbol)

    results["recall"] = (
        results["true_positives"] / results["ground_truth"]
        if results["ground_truth"]
        else 0.0
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        missed_classes = ", ".join(results["missed_classes"]) or "none"

        markdown_path.write_text(
            f"# {args.model} benchmark\n\n"
            f"- Dataset: `{dataset_dir.name}`\n"
            f"- Benchmark: `{benchmark}`\n"
            f"- Model: `{args.model_id}`\n"
            f"- Evaluation domain: `{', '.join(results['evaluation_domain'])}`\n"
            f"- IoU threshold: `{args.iou_threshold}`\n"
            f"- Confidence threshold: `{args.confidence}`\n"
            f"- In-domain ground truth: `{results['ground_truth']}`\n"
            f"- Out-of-domain ground truth: `{results['out_of_domain_ground_truth']}`\n"
            f"- Background images: `{results['background_images']}`\n"
            f"- Background false positives: `{results['background_false_positives']}`\n"
            f"- True positives: `{results['true_positives']}`\n"
            f"- False positives: `{results['false_positives']}`\n"
            f"- Recall: `{results['recall']:.2%}`\n"
            f"- Missed classes: `{missed_classes}`\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()