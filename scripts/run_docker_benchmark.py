#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "ajdanoff/asl-gesture-recognition"

BENCHMARKS = {
    "original": {
        "dataset": "ajdanoff/asl-webcam-benchmark-original",
        "directory": "benchmark/asl-webcam-benchmark-original-v1.0.0",
        "expected_images": 36,
    },
    "augmented": {
        "dataset": "ajdanoff/asl-webcam-benchmark-augmented",
        "directory": "benchmark/asl-webcam-benchmark-augmented-v1.0.0",
        "expected_images": 216,
    },
}

MODELS = {
    "unified": {
        "model_id": "ajdanoff/asl-unified-webcam-yolo11n-mp",
        "revision": "d3d43f88b6e03cb6f2d8a9eb31a0c92e1e61197b",
    },
    "digits": {
        "model_id": "ajdanoff/asl-unified-digits-yolo11n-mp",
        "revision": "34ba348ac8064ee1fd270146e85ee3853be48495",
    },
    "letters": {
        "model_id": "ajdanoff/asl-unified-letters-yolo11n-mp",
        "revision": "e81224f2c93dfc592815b45de914fe3ebc9a3425",
    },
}


def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}")


def download_benchmark(benchmark, dataset, directory):
    Path(directory).parent.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "hf",
            "download",
            dataset,
            "--repo-type",
            "dataset",
            "--local-dir",
            directory,
        ]
    )

    return Path(directory)


def evaluate_model(
    benchmark,
    dataset_dir,
    model_name,
    model_id,
    model_revision,
    iou_threshold,
    confidence,
    output_dir,
):
    data_yaml = dataset_dir / "data.yaml"
    result_path = output_dir / f"{model_name}.json"
    markdown_path = output_dir / f"{model_name}.md"

    run_command(
        [
            sys.executable,
            "scripts/run_benchmark.py",
            "--dataset-dir",
            str(dataset_dir),
            "--data-yaml",
            str(data_yaml),
            "--model",
            model_name,
            "--model-id",
            model_id,
            "--model-revision",
            model_revision,
            "--benchmark",
            benchmark,
            "--iou-threshold",
            str(iou_threshold),
            "--confidence",
            str(confidence),
            "--output",
            str(result_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    with result_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_missed_classes(missed_classes):
    return ", ".join(missed_classes) if missed_classes else "none"


def generate_extended_report(
    benchmark,
    results,
    evaluator_commit,
    iou_threshold,
    confidence,
    output_path,
):
    generated_at = datetime.now(timezone.utc).isoformat()

    lines = [
        f"# ASL Webcam Benchmark Report — {benchmark.title()}",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This report evaluates YOLO11n ASL gesture detectors on a held-out",
        f"`{benchmark}` webcam benchmark using symbol-based matching.",
        "",
        "A prediction is counted as a true positive only when its decoded symbol",
        "matches the ground-truth symbol and its IoU with the matching",
        f"ground-truth box is at least `{iou_threshold}`.",
        "",
        "## Evaluation configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Benchmark | `{benchmark}` |",
        f"| Evaluator commit | [`{evaluator_commit[:7]}`]({REPOSITORY}/commit/{evaluator_commit}) |",
        f"| Confidence threshold | `{confidence}` |",
        f"| IoU threshold | `{iou_threshold}` |",
        f"| Matching rule | Symbol matching |",
        f"| Generated at | `{generated_at}` |",
        "",
        "## Aggregate results",
        "",
        "| Model | Evaluation domain | In-domain ground truth | True positives | Recall | Background images | Background false positives |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for model_name, result in results.items():
        evaluation_domain = ", ".join(result["evaluation_domain"])
        background_false_positives = result.get(
            "background_false_positives",
            0,
        )
        background_images = result.get("background_images", 0)

        lines.append(
            f"| {model_name.title()} | `{evaluation_domain}` | "
            f"{result['ground_truth']} | {result['true_positives']} | "
            f"{result['recall']:.2%} | {background_images} | "
            f"{background_false_positives} |"
        )

    lines.extend(
        [
            "",
            "## Per-class results",
            "",
        ]
    )

    for model_name, result in results.items():
        lines.extend(
            [
                f"### {model_name.title()} model",
                "",
                f"Model: [`{result['model_id']}`]({result['model_id']})",
                "",
                "| Class | Ground truth | True positives | Recall |",
                "|---|---:|---:|---:|",
            ]
        )

        for symbol, stats in sorted(result["per_class"].items()):
            lines.append(
                f"| `{symbol}` | {stats['ground_truth']} | "
                f"{stats['true_positives']} | {stats['recall']:.2%} |"
            )

        missed_classes = format_missed_classes(result["missed_classes"])

        lines.extend(
            [
                "",
                f"**Missed classes:** `{missed_classes}`",
                "",
                f"**False positives:** `{result['false_positives']}`",
                "",
                f"**Out-of-domain ground truth:** "
                f"`{result.get('out_of_domain_ground_truth', 0)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## False-positive diagnostics",
            "",
        ]
    )

    has_details = False

    for model_name, result in results.items():
        details = result.get("false_positive_details", [])

        if not details:
            continue

        has_details = True

        lines.extend(
            [
                f"### {model_name.title()} model",
                "",
                "| Image | Predicted symbol | Confidence | Box (x1, y1, x2, y2) |",
                "|---|---|---:|---|",
            ]
        )

        for detail in details:
            for prediction in detail["predictions"]:
                box = prediction["box_xyxy"]
                box_text = (
                    f"({box[0]:.1f}, {box[1]:.1f}, "
                    f"{box[2]:.1f}, {box[3]:.1f})"
                )

                lines.append(
                    f"| `{detail['image']}` | `{prediction['symbol']}` | "
                    f"{prediction['confidence']:.3f} | `{box_text}` |"
                )

        lines.append("")

    if not has_details:
        lines.append("No false-positive details were recorded.")

    lines.extend(
        [
            "## Interpretation",
            "",
            "The lower augmented-benchmark recall indicates that the current",
            "augmentation pipeline is more challenging than the models’ training",
            "and validation distributions.",
            "",
            "Specialist models are evaluated only within their own class domain.",
            "Therefore, digit-only recall uses only digit ground truth, while",
            "letter-only recall uses only letter ground truth.",
            "",
            "The benchmark contains a small number of original source images, so",
            "results should be interpreted as failure-mode diagnostics rather than",
            "a definitive production-model comparison.",
            "",
            "## Limitations",
            "",
            "- The original benchmark contains only one image per gesture class.",
            "- All augmented variants derive from the same 36 source webcam images.",
            "- The benchmark is not yet a large multi-session held-out test set.",
            "- Results should not be used alone to select a final production model.",
            "- Background-image counts are too small for strong background-suppression",
            "  conclusions.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete ASL webcam benchmark in Docker."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["original", "augmented"],
    )
    parser.add_argument(
        "--models",
        default="unified,digits,letters",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark-reports",
    )
    args = parser.parse_args()

    benchmark_config = BENCHMARKS[args.benchmark]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator_commit = os.environ.get(
        "GITHUB_SHA",
        "local-development",
    )

    dataset_dir = download_benchmark(
        args.benchmark,
        benchmark_config["dataset"],
        benchmark_config["directory"],
    )

    run_command(
        [
            sys.executable,
            "scripts/validate_benchmark.py",
            "--dataset-dir",
            str(dataset_dir),
            "--data-yaml",
            str(dataset_dir / "data.yaml"),
            "--benchmark",
            args.benchmark,
        ]
    )

    selected_models = [
        model.strip()
        for model in args.models.split(",")
        if model.strip()
    ]

    results = {}

    for model_name in selected_models:
        if model_name not in MODELS:
            raise SystemExit(f"Unknown model: {model_name}")

        model_config = MODELS[model_name]

        results[model_name] = evaluate_model(
            args.benchmark,
            dataset_dir,
            model_name,
            model_config["model_id"],
            model_config["revision"],
            args.iou_threshold,
            args.confidence,
            output_dir,
        )

    report_path = output_dir / f"benchmark-report-{args.benchmark}.md"

    generate_extended_report(
        args.benchmark,
        results,
        evaluator_commit,
        args.iou_threshold,
        args.confidence,
        report_path,
    )

    print(f"Extended report generated: {report_path}")


if __name__ == "__main__":
    main()