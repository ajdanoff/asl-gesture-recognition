from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import mediapipe as mp


ALL_CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]
DIGIT_NAMES = [str(index) for index in range(10)]
LETTER_NAMES = [
    class_name
    for class_name in ALL_CLASS_NAMES
    if class_name.isalpha()
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert source images into a YOLO detection dataset using "
            "MediaPipe boxes, optional class-matched YOLO fallback boxes, "
            "and class/image-size-aware centered fallback boxes."
        )
    )

    parser.add_argument(
        "--source-1",
        type=Path,
        default=Path("asl-alphabet-dataset"),
        help="Root of the first source dataset.",
    )
    parser.add_argument(
        "--source-2",
        type=Path,
        default=Path("asl_unified_dataset"),
        help="Root of the second source dataset.",
    )
    parser.add_argument(
        "--webcam-root",
        type=Path,
        default=Path("webcam_data"),
        help=(
            "Root directory containing webcam_gestures and webcam_bg "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/asl_digits_mp"),
        help=(
            "Output dataset directory. It will be deleted if it already "
            "exists."
        ),
    )
    parser.add_argument(
        "--classes",
        choices=("digits", "letters", "all"),
        required=True,
        help=(
            "Classes to include: digits (0-9), letters (A-Z), or all "
            "(digits and letters)."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images assigned to the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for shuffling and deterministic train/val "
            "assignment."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Write visualization images showing boxes and their sources."
        ),
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help=(
            "Include webcam_bg images as unlabeled negative examples. "
            "No label file is written for these images."
        ),
    )
    parser.add_argument(
        "--hand-landmarker",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to the MediaPipe HandLandmarker task file.",
    )
    parser.add_argument(
        "--fallback-box",
        choices=("running-mean", "skip"),
        default="running-mean",
        help=(
            "Policy when neither MediaPipe nor YOLO fallback produces a "
            "box: use a class/image-size-aware centered mean box, or skip "
            "the image."
        ),
    )
    parser.add_argument(
        "--fallback-yolo-model",
        type=Path,
        help=(
            "Optional YOLO model used when MediaPipe fails. A detection is "
            "accepted only if its predicted class matches the expected "
            "class."
        ),
    )
    parser.add_argument(
        "--fallback-yolo-conf",
        type=float,
        default=0.50,
        help="Minimum confidence for accepting a YOLO fallback detection.",
    )
    parser.add_argument(
        "--fallback-yolo-iou",
        type=float,
        default=0.45,
        help="IoU threshold used during YOLO fallback inference.",
    )
    parser.add_argument(
        "--fallback-margin",
        type=float,
        default=1.25,
        help=(
            "Width and height expansion factor applied to class/image-size "
            "aware centered fallback boxes. Use 1.0 for no expansion."
        ),
    )
    parser.add_argument(
        "--size-bins",
        type=int,
        default=4,
        help=(
            "Number of equal-count image-size bins used for fallback "
            "statistics. Bins are based on image area."
        ),
    )
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=10,
        help=(
            "Minimum number of successful MediaPipe detections required in "
            "a (class, size-bin) group before using its mean."
        ),
    )
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=20,
        help=(
            "Minimum number of successful MediaPipe detections required in "
            "a class before using a class-only mean instead of the global "
            "mean."
        ),
    )

    return parser.parse_args()


def selected_class_names(classes: str) -> list[str]:
    if classes == "digits":
        return DIGIT_NAMES.copy()

    if classes == "letters":
        return LETTER_NAMES.copy()

    return ALL_CLASS_NAMES.copy()


def get_source_1_samples(
    root: Path,
    allowed_classes: set[str],
) -> list[tuple[str, str, Path]]:
    """
    Source 1 contains alphabet images only.
    """
    samples = []

    for class_name in LETTER_NAMES:
        if class_name not in allowed_classes:
            continue

        class_folder = root / f"{class_name}-samples"

        if not class_folder.is_dir():
            continue

        for image_path in sorted(class_folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    ("source1", class_name, image_path)
                )

    return samples


def get_source_2_samples(
    root: Path,
    allowed_classes: set[str],
) -> list[tuple[str, str, Path]]:
    samples = []

    for class_name in ALL_CLASS_NAMES:
        if class_name not in allowed_classes:
            continue

        class_folder = root / class_name.lower()

        if not class_folder.is_dir():
            continue

        for image_path in sorted(class_folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    ("source2", class_name, image_path)
                )

    return samples


def get_webcam_gesture_samples(
    root: Path,
    allowed_classes: set[str],
) -> list[tuple[str, str, Path]]:
    samples = []
    gestures_root = root / "webcam_gestures"

    if not gestures_root.is_dir():
        return samples

    for class_name in ALL_CLASS_NAMES:
        if class_name not in allowed_classes:
            continue

        class_folder = gestures_root / class_name

        if not class_folder.is_dir():
            continue

        for image_path in sorted(class_folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    ("webcam", class_name, image_path)
                )

    return samples


def get_webcam_background_images(
    root: Path,
) -> list[tuple[str, None, Path]]:
    samples = []
    background_root = root / "webcam_bg"

    if not background_root.is_dir():
        return samples

    for image_path in sorted(background_root.iterdir()):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            samples.append(("webcam_bg", None, image_path))

    return samples


def get_all_samples(
    source_1: Path,
    source_2: Path,
    webcam_root: Path,
    allowed_classes: set[str],
) -> list[tuple[str, str, Path]]:
    samples = []

    samples.extend(
        get_source_1_samples(source_1, allowed_classes)
    )
    samples.extend(
        get_source_2_samples(source_2, allowed_classes)
    )
    samples.extend(
        get_webcam_gesture_samples(webcam_root, allowed_classes)
    )

    return samples


def calculate_equal_count_bin_edges(
    values: list[float],
    number_of_bins: int,
) -> list[float]:
    """
    Calculate internal bin edges using equal-count quantiles.

    Returns number_of_bins - 1 internal edges. This creates bins with
    approximately equal numbers of samples, rather than bins with equal
    value ranges.
    """
    if number_of_bins < 1:
        raise ValueError("Number of size bins must be at least 1")

    if len(values) < number_of_bins:
        raise ValueError(
            "Not enough values to create the requested number of bins"
        )

    sorted_values = sorted(values)
    edges = []

    for index in range(1, number_of_bins):
        position = index * len(sorted_values) / number_of_bins
        lower_index = int(position)
        upper_index = min(
            lower_index + 1,
            len(sorted_values) - 1,
        )
        fraction = position - lower_index

        edge = (
            sorted_values[lower_index] * (1.0 - fraction)
            + sorted_values[upper_index] * fraction
        )

        edges.append(edge)

    return edges


def get_size_bin_key(
    width: int,
    height: int,
) -> str:
    """
    Return a stable source-resolution group key.

    Known groups in the letters dataset:
      - 400x400 source images
      - 224x224 source images
      - 640x480 webcam images
    """
    if width == 400 and height == 400:
        return "source_400x400"

    if width == 224 and height == 224:
        return "source_224x224"

    if width == 640 and height == 480:
        return "webcam_640x480"

    # Preserve useful fallback groups for unexpected resolutions.
    aspect_ratio = width / height

    if abs(aspect_ratio - 1.0) <= 0.05:
        return "other_square"

    if abs(aspect_ratio - 4 / 3) <= 0.05:
        return "other_4_3"

    return f"other_{width}x{height}"

def find_hand_box(
    image_bgr,
    landmarker,
) -> tuple[int, int, int, int] | None:
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    x_values = [landmark.x * width for landmark in landmarks]
    y_values = [landmark.y * height for landmark in landmarks]

    x_min = max(0.0, min(x_values))
    y_min = max(0.0, min(y_values))
    x_max = min(float(width - 1), max(x_values))
    y_max = min(float(height - 1), max(y_values))

    box_width = max(1.0, x_max - x_min)
    box_height = max(1.0, y_max - y_min)

    margin_x = 0.12 * box_width
    margin_y = 0.12 * box_height

    x1 = max(0, int(x_min - margin_x))
    y1 = max(0, int(y_min - margin_y))
    x2 = min(width - 1, int(x_max + margin_x))
    y2 = min(height - 1, int(y_max + margin_y))

    return x1, y1, x2, y2


def find_yolo_fallback_box(
    image_bgr,
    yolo_model,
    expected_class_name: str,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int, int] | None:
    """
    Return the highest-confidence YOLO box only when its predicted class
    matches the expected class.
    """
    results = yolo_model.predict(
        image_bgr,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return None

    result = results[0]
    boxes = result.boxes

    best_index = None
    best_confidence = 0.0

    for index, class_id in enumerate(boxes.cls.tolist()):
        predicted_class_name = yolo_model.names[int(class_id)]

        if predicted_class_name != expected_class_name:
            continue

        confidence = float(boxes.conf[index])

        if confidence > best_confidence:
            best_confidence = confidence
            best_index = index

    if best_index is None:
        return None

    x1, y1, x2, y2 = boxes.xyxy[best_index].tolist()
    height, width = image_bgr.shape[:2]

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(width - 1, int(round(x2)))
    y2 = min(height - 1, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def centered_box_from_ratios(
    width: int,
    height: int,
    mean_width_ratio: float,
    mean_height_ratio: float,
    margin: float,
) -> tuple[int, int, int, int]:
    """
    Create a centered pixel box using normalized mean width/height ratios
    and an expansion margin.
    """
    expanded_width_ratio = mean_width_ratio * margin
    expanded_height_ratio = mean_height_ratio * margin

    box_width = max(
        1,
        min(width, int(round(expanded_width_ratio * width))),
    )
    box_height = max(
        1,
        min(height, int(round(expanded_height_ratio * height))),
    )

    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    return x1, y1, x2, y2


def get_fallback_ratios(
    class_name: str,
    size_bin: str,
    class_size_statistics: dict[
        tuple[str, str],
        dict[str, list[float]],
    ],
    class_mean_width: dict[str, float],
    class_mean_height: dict[str, float],
    global_mean_width: float,
    global_mean_height: float,
    min_bin_samples: int,
) -> tuple[float, float, str]:
    """
    Return normalized fallback width/height ratios and the fallback level.

    Fallback hierarchy:
      1. class + image-size bin
      2. class only
      3. global mean
    """
    statistics = class_size_statistics.get(
        (class_name, size_bin)
    )

    if (
        statistics
        and len(statistics["width"]) >= min_bin_samples
        and len(statistics["height"]) >= min_bin_samples
    ):
        return (
            sum(statistics["width"])
            / len(statistics["width"]),
            sum(statistics["height"])
            / len(statistics["height"]),
            "class-size",
        )

    return (
        class_mean_width.get(
            class_name,
            global_mean_width,
        ),
        class_mean_height.get(
            class_name,
            global_mean_height,
        ),
        "class-only",
    )


def to_yolo_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height

    return (
        x_center,
        y_center,
        box_width / width,
        box_height / height,
    )


def output_name(
    source_name: str,
    class_name: str | None,
    image_path: Path,
    index: int,
) -> str:
    if class_name is None:
        return f"{source_name}_bg_{index:06d}.jpg"

    return (
        f"{source_name}_{class_name}_{image_path.stem}.jpg"
    )


def write_yaml(
    output_root: Path,
    class_names: list[str],
) -> None:
    names = ", ".join(
        f"'{class_name}'" for class_name in class_names
    )
    yaml_path = output_root / "data.yaml"

    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def save_image(
    image,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")

def main():
    args = parse_args()
    random.seed(args.seed)

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(
            "--val-ratio must be between 0 and 1"
        )

    if args.fallback_margin <= 0.0:
        raise ValueError(
            "--fallback-margin must be greater than 0"
        )

    if args.size_bins < 1:
        raise ValueError("--size-bins must be at least 1")

    if args.min_bin_samples < 1:
        raise ValueError(
            "--min-bin-samples must be at least 1"
        )

    if args.min_class_samples < 1:
        raise ValueError(
            "--min-class-samples must be at least 1"
        )

    for path in (
        args.source_1,
        args.source_2,
        args.webcam_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)

    if not args.hand_landmarker.is_file():
        raise FileNotFoundError(args.hand_landmarker)

    fallback_yolo_model = None

    if args.fallback_yolo_model:
        if not HAS_ULTRALYTICS:
            raise ImportError(
                "Install ultralytics to use --fallback-yolo-model."
            )

        if not args.fallback_yolo_model.is_file():
            raise FileNotFoundError(
                args.fallback_yolo_model
            )

        fallback_yolo_model = YOLO(
            str(args.fallback_yolo_model)
        )

    # Derive supported fallback classes from the loaded YOLO model.
    # This automatically handles models with different class vocabularies.
    if fallback_yolo_model is not None:
        yolo_fallback_classes = set(
            fallback_yolo_model.names.values()
        )
    else:
        yolo_fallback_classes = set()

    class_names = selected_class_names(args.classes)
    class_to_id = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }
    allowed_classes = set(class_names)

    if args.output.exists():
        shutil.rmtree(args.output)

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (args.output / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        if args.visualize:
            (args.output / "vis" / split).mkdir(
                parents=True,
                exist_ok=True,
            )

    labeled_samples = get_all_samples(
        args.source_1,
        args.source_2,
        args.webcam_root,
        allowed_classes,
    )
    random.shuffle(labeled_samples)

    background_images = (
        get_webcam_background_images(args.webcam_root)
        if args.include_background
        else []
    )
    random.shuffle(background_images)

    # Assign train/val splits before processing. This keeps split
    # assignment stable even if some images are skipped later.
    labeled_splits = [
        "val" if random.random() < args.val_ratio else "train"
        for _ in labeled_samples
    ]

    background_splits = [
        "val" if random.random() < args.val_ratio else "train"
        for _ in background_images
    ]

    counts = {
        "train": {
            "images": 0,
            "background": 0,
        },
        "val": {
            "images": 0,
            "background": 0,
        },
    }

    class_counts = {
        split: {
            class_name: 0
            for class_name in class_names
        }
        for split in ("train", "val")
    }

    box_source_counts = {
        "mediapipe": 0,
        "yolo_fallback": 0,
        "centered_fallback": 0,
    }

    fallback_count = 0
    skipped_fallback_count = 0
    unreadable_count = 0

    # Pre-analysis stage: determine image-size bins.
    print("\nPre-analysis: image-size distribution")

    sample_size_bin_keys: list[str] = []
    size_bin_counts: dict[str, int] = {}

    for _, _, image_path in labeled_samples:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            unreadable_count += 1
            sample_size_bin_keys.append("unreadable")
            continue

        height, width = image.shape[:2]
        size_bin = get_size_bin_key(width, height)

        sample_size_bin_keys.append(size_bin)
        size_bin_counts[size_bin] = (
            size_bin_counts.get(size_bin, 0) + 1
        )

    print("Image-size groups:")
    for size_bin, count in sorted(size_bin_counts.items()):
        print(f"  {size_bin}: {count} images")

    if len(size_bin_counts) < 2:
        print(
            "Warning: only one image-size group was found. "
            "Class/image-size fallback statistics will behave like "
            "class-only statistics."
        )

    # First pass: obtain boxes and accumulate class/size-bin statistics.
    box_cache: dict[
        int,
        tuple[
            tuple[int, int, int, int],
            str,
        ]
        | None
    ] = {}

    detected_box_count = 0
    global_width_ratio_sum = 0.0
    global_height_ratio_sum = 0.0

    class_size_statistics: dict[
        tuple[str, str],
        dict[str, list[float]],
    ] = {}

    hand_landmarker_options = (
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(
                    args.hand_landmarker
                ),
            ),
            running_mode=(
                mp.tasks.vision.RunningMode.IMAGE
            ),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    with (
        mp.tasks.vision.HandLandmarker.create_from_options(
            hand_landmarker_options
        ) as landmarker
    ):
        print("\nFirst pass: detecting hand boxes")

        for sample_index, (
            source_name,
            class_name,
            image_path,
        ) in enumerate(labeled_samples):
            image = cv2.imread(str(image_path))

            if image is None:
                unreadable_count += 1
                box_cache[sample_index] = None
                continue

            height, width = image.shape[:2]

            # Primary source: MediaPipe.
            box = find_hand_box(image, landmarker)
            box_source = "mediapipe"

            # Fallback: class-matched YOLO box.
            if box is None:
                if (
                    fallback_yolo_model is not None
                    and class_name in yolo_fallback_classes
                ):
                    box = find_yolo_fallback_box(
                        image,
                        fallback_yolo_model,
                        class_name,
                        args.fallback_yolo_conf,
                        args.fallback_yolo_iou,
                    )

                    if box is not None:
                        box_source = "yolo_fallback"

            if box is None:
                box_cache[sample_index] = None
                continue

            x1, y1, x2, y2 = box

            normalized_width = (
                max(1, x2 - x1) / width
            )
            normalized_height = (
                max(1, y2 - y1) / height
            )

            # Only MediaPipe boxes contribute to the fallback-size
            # distribution. This prevents YOLO-generated boxes from
            # influencing the estimate used when both detectors fail.
            if box_source == "mediapipe":
                size_bin = sample_size_bin_keys[
                    sample_index
                ]
                statistics_key = (
                    class_name,
                    size_bin,
                )

                if (
                    statistics_key
                    not in class_size_statistics
                ):
                    class_size_statistics[
                        statistics_key
                    ] = {
                        "width": [],
                        "height": [],
                    }

                class_size_statistics[
                    statistics_key
                ]["width"].append(normalized_width)
                class_size_statistics[
                    statistics_key
                ]["height"].append(normalized_height)

                global_width_ratio_sum += (
                    normalized_width
                )
                global_height_ratio_sum += (
                    normalized_height
                )
                detected_box_count += 1

            box_cache[sample_index] = (
                box,
                box_source,
            )

        if detected_box_count == 0:
            raise RuntimeError(
                "No MediaPipe hand boxes were detected. Cannot calculate "
                "fallback box dimensions."
            )

        global_mean_width = (
            global_width_ratio_sum / detected_box_count
        )
        global_mean_height = (
            global_height_ratio_sum
            / detected_box_count
        )

        # Calculate class-only means.
        class_width_ratios: dict[
            str, list[float]
        ] = {
            class_name: []
            for class_name in class_names
        }
        class_height_ratios: dict[
            str, list[float]
        ] = {
            class_name: []
            for class_name in class_names
        }

        for (
            (class_name, _),
            statistics,
        ) in class_size_statistics.items():
            class_width_ratios[class_name].extend(
                statistics["width"]
            )
            class_height_ratios[class_name].extend(
                statistics["height"]
            )

        class_mean_width: dict[str, float] = {}
        class_mean_height: dict[str, float] = {}

        for class_name in class_names:
            width_ratios = class_width_ratios[
                class_name
            ]
            height_ratios = class_height_ratios[
                class_name
            ]

            enough_samples = (
                len(width_ratios)
                >= args.min_class_samples
                and len(height_ratios)
                >= args.min_class_samples
            )

            if enough_samples:
                class_mean_width[class_name] = (
                    sum(width_ratios)
                    / len(width_ratios)
                )
                class_mean_height[class_name] = (
                    sum(height_ratios)
                    / len(height_ratios)
                )
            else:
                class_mean_width[class_name] = (
                    global_mean_width
                )
                class_mean_height[class_name] = (
                    global_mean_height
                )

        print("\nFallback statistics")
        print(
            "Global mean normalized size: "
            f"width={global_mean_width:.4f}, "
            f"height={global_mean_height:.4f}"
        )

        for class_name in class_names:
            width_count = len(
                class_width_ratios[class_name]
            )

            if (
                width_count
                >= args.min_class_samples
            ):
                print(
                    f"  {class_name}: "
                    f"detected={width_count}, "
                    f"mean_width="
                    f"{class_mean_width[class_name]:.4f}, "
                    f"mean_height="
                    f"{class_mean_height[class_name]:.4f}"
                )
            else:
                print(
                    f"  {class_name}: "
                    f"detected={width_count}; "
                    "global mean used for fallback"
                )

        # Second pass: write dataset using cached boxes.
        print("\nSecond pass: writing dataset")

        for sample_index, (
            source_name,
            class_name,
            image_path,
        ) in enumerate(labeled_samples):
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            height, width = image.shape[:2]
            cached_box = box_cache[sample_index]

            if cached_box is None:
                fallback_count += 1

                if args.fallback_box == "skip":
                    skipped_fallback_count += 1
                    continue

                size_bin = sample_size_bin_keys[
                    sample_index
                ]

                (
                    fallback_width_ratio,
                    fallback_height_ratio,
                    fallback_level,
                ) = get_fallback_ratios(
                    class_name,
                    size_bin,
                    class_size_statistics,
                    class_mean_width,
                    class_mean_height,
                    global_mean_width,
                    global_mean_height,
                    args.min_bin_samples,
                )

                box = centered_box_from_ratios(
                    width,
                    height,
                    fallback_width_ratio,
                    fallback_height_ratio,
                    args.fallback_margin,
                )
                box_source = "centered_fallback"
            else:
                box, box_source = cached_box
                fallback_level = None

            split = labeled_splits[sample_index]
            filename = output_name(
                source_name,
                class_name,
                image_path,
                sample_index,
            )

            out_image_path = (
                args.output
                / "images"
                / split
                / filename
            )
            out_label_path = (
                args.output
                / "labels"
                / split
                / f"{Path(filename).stem}.txt"
            )

            save_image(image, out_image_path)

            x_center, y_center, box_width, box_height = (
                to_yolo_box(box, width, height)
            )

            class_id = class_to_id[class_name]

            out_label_path.write_text(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n",
                encoding="utf-8",
            )

            if args.visualize:
                visualization = image.copy()
                x1, y1, x2, y2 = box

                if box_source == "mediapipe":
                    color = (0, 255, 0)  # Green
                    label = class_name
                elif box_source == "yolo_fallback":
                    color = (255, 0, 0)  # Blue in BGR
                    label = f"{class_name} YOLO-FB"
                else:
                    color = (0, 165, 255)  # Orange
                    label = (
                        f"{class_name} "
                        f"FB-{fallback_level}"
                    )

                cv2.rectangle(
                    visualization,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    visualization,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                out_visualization_path = (
                    args.output
                    / "vis"
                    / split
                    / filename
                )
                save_image(
                    visualization,
                    out_visualization_path,
                )

            counts[split]["images"] += 1
            class_counts[split][class_name] += 1
            box_source_counts[box_source] += 1

        # Background images: no labels.
        for background_index, (
            source_name,
            _,
            image_path,
        ) in enumerate(background_images):
            image = cv2.imread(str(image_path))

            if image is None:
                unreadable_count += 1
                continue

            split = background_splits[
                background_index
            ]
            filename = output_name(
                source_name,
                None,
                image_path,
                background_index,
            )

            out_image_path = (
                args.output
                / "images"
                / split
                / filename
            )
            save_image(image, out_image_path)

            # No label file is written: YOLO treats this as no objects.

            if args.visualize:
                visualization = image.copy()

                cv2.putText(
                    visualization,
                    "BACKGROUND",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                out_visualization_path = (
                    args.output
                    / "vis"
                    / split
                    / filename
                )
                save_image(
                    visualization,
                    out_visualization_path,
                )

            counts[split]["images"] += 1
            counts[split]["background"] += 1

    write_yaml(args.output, class_names)

    class_size_manifest = {}

    for (
        (class_name, size_bin),
        statistics,
    ) in class_size_statistics.items():
        key = f"{class_name}|{size_bin}"
        width_values = statistics["width"]
        height_values = statistics["height"]

        class_size_manifest[key] = {
            "width_count": len(width_values),
            "height_count": len(height_values),
            "mean_width": (
                sum(width_values) / len(width_values)
                if width_values
                else None
            ),
            "mean_height": (
                sum(height_values) / len(height_values)
                if height_values
                else None
            ),
        }

    manifest = {
        "classes_mode": args.classes,
        "class_names": class_names,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "include_background": args.include_background,
        "fallback_box_policy": args.fallback_box,
        "fallback_margin": args.fallback_margin,
        "size_bin_strategy": "explicit_resolution_groups",
        "min_bin_samples": args.min_bin_samples,
        "min_class_samples": args.min_class_samples,
        "fallback_yolo_model": (
            str(args.fallback_yolo_model)
            if args.fallback_yolo_model
            else None
        ),
        "fallback_yolo_conf": args.fallback_yolo_conf,
        "fallback_yolo_iou": args.fallback_yolo_iou,
        "hand_landmarker": str(args.hand_landmarker),
        "size_bin_counts": size_bin_counts,
        "global_mean_width": global_mean_width,
        "global_mean_height": global_mean_height,
        "class_mean_width": class_mean_width,
        "class_mean_height": class_mean_height,
        "class_size_statistics": class_size_manifest,
        "detected_media_pipe_boxes": detected_box_count,
        "box_source_counts": box_source_counts,
        "fallback_images_considered": fallback_count,
        "fallback_images_skipped": skipped_fallback_count,
        "unreadable_images_skipped": unreadable_count,
        "counts": counts,
        "class_counts": class_counts,
    }

    manifest_path = (
        args.output / "conversion_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nConversion complete")
    print(f"Class mode: {args.classes}")
    print(
        f"Classes ({len(class_names)}): "
        f"{', '.join(class_names)}"
    )
    print(f"Train images: {counts['train']['images']}")
    print(
        f"Validation images: {counts['val']['images']}"
    )
    print(
        "Train background images: "
        f"{counts['train']['background']}"
    )
    print(
        "Validation background images: "
        f"{counts['val']['background']}"
    )
    print(
        f"Unreadable images skipped: {unreadable_count}"
    )

    print("\nBox sources")
    print(
        f"MediaPipe: {box_source_counts['mediapipe']}"
    )
    print(
        "YOLO fallback: "
        f"{box_source_counts['yolo_fallback']}"
    )
    print(
        "Centered fallback: "
        f"{box_source_counts['centered_fallback']}"
    )
    print(
        "Images skipped due to failed fallback: "
        f"{skipped_fallback_count}"
    )

    print("\nInstances by class")
    for class_name in class_names:
        print(
            f"  {class_name}: "
            f"train={class_counts['train'][class_name]}, "
            f"val={class_counts['val'][class_name]}"
        )

    print(f"\nOutput: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()