#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []

    for result_path in sorted(results_dir.glob("*.json")):
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

        rows.append(
            (
                result["model"],
                result["ground_truth"],
                result["true_positives"],
                result["recall"],
                result["missed_classes"],
            )
        )

    lines = [
        "# Benchmark summary",
        "",
        "| Model | Ground truth | True positives | Recall | Missed classes |",
        "|---|---:|---:|---:|---|",
    ]

    for model, ground_truth, true_positives, recall, missed in rows:
        lines.append(
            f"| {model} | {ground_truth} | {true_positives} | "
            f"{recall:.2%} | {', '.join(missed) or 'none'} |"
        )

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()