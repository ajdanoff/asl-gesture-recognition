from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate unified, digit-only, and letter-only YOLO models "
            "on the same held-out webcam test set."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to shared test dataset data.yaml.",
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
        help="Validation image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Minimum confidence used during validation.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold used during validation.",
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

    return parser.parse_args()


def evaluate_model(
    model_weights: Path,
    data_yaml: Path,
    args,
    run_name: str,
):
    model = YOLO(str(model_weights))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
        project=str(args.output),
        name=run_name,
        exist_ok=False,
    )

    results = {
        "model": str(model_weights),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_ap50": {
            model.names[index]: float(value)
            for index, value in enumerate(metrics.box.ap50)
        },
        "per_class_ap50_95": {
            model.names[index]: float(value)
            for index, value in enumerate(metrics.box.ap)
        },
    }

    report_path = args.output / run_name / "summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\n{run_name}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"mAP@0.50: {results['map50']:.4f}")
    print(f"mAP@0.50:0.95: {results['map50_95']:.4f}")
    print(f"Report: {report_path}")

    return results


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results = {}

    results["unified"] = evaluate_model(
        model_weights=args.unified_model,
        data_yaml=args.data,
        args=args,
        run_name="unified_model",
    )

    results["digits"] = evaluate_model(
        model_weights=args.digits_model,
        data_yaml=args.data,
        args=args,
        run_name="digits_model",
    )

    results["letters"] = evaluate_model(
        model_weights=args.letters_model,
        data_yaml=args.data,
        args=args,
        run_name="letters_model",
    )

    report_path = args.output / "shared_webcam_test_summary.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nCombined report: {report_path}")


if __name__ == "__main__":
    main()