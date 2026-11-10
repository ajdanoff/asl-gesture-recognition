from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an augmented shared webcam test dataset from one image "
            "or a folder of images. Optionally transforms YOLO bounding "
            "boxes together with the images."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input image file or folder containing images. If a label file "
            "with the same stem exists, its YOLO boxes are transformed too."
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help=(
            "Optional folder containing YOLO label files. If omitted, the "
            "script looks beside each input image."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for augmented images and labels.",
    )
    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=5,
        help=(
            "Number of augmented variants to generate for each input image. "
            "The original image is also copied."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=640,
        help="Final output image width.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=480,
        help="Final output image height.",
    )
    parser.add_argument(
        "--no-horizontal-flip",
        action="store_true",
        help="Disable horizontal flipping.",
    )
    parser.add_argument(
        "--horizontal-flip-probability",
        type=float,
        default=0.5,
        help="Probability of horizontal flip for each generated variant.",
    )
    parser.add_argument(
        "--crop-min-area",
        type=float,
        default=0.80,
        help=(
            "Minimum fraction of the original image area retained by a "
            "random crop, before final resizing."
        ),
    )
    parser.add_argument(
        "--crop-max-area",
        type=float,
        default=1.00,
        help=(
            "Maximum fraction of the original image area retained by a "
            "random crop, before final resizing."
        ),
    )
    parser.add_argument(
        "--rotation-max-degrees",
        type=float,
        default=12.0,
        help="Maximum absolute random rotation angle in degrees.",
    )
    parser.add_argument(
        "--translation-max-ratio",
        type=float,
        default=0.08,
        help=(
            "Maximum absolute random translation, as a fraction of image "
            "width or height."
        ),
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.85,
        help="Minimum random scale factor.",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.15,
        help="Maximum random scale factor.",
    )
    parser.add_argument(
        "--brightness-max-change",
        type=float,
        default=0.20,
        help="Maximum relative brightness change.",
    )
    parser.add_argument(
        "--contrast-max-change",
        type=float,
        default=0.20,
        help="Maximum relative contrast change.",
    )
    parser.add_argument(
        "--gamma-min",
        type=float,
        default=0.75,
        help="Minimum gamma correction value.",
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=1.35,
        help="Maximum gamma correction value.",
    )
    parser.add_argument(
        "--hue-max-change",
        type=int,
        default=8,
        help="Maximum absolute OpenCV hue shift.",
    )
    parser.add_argument(
        "--saturation-max-change",
        type=float,
        default=0.20,
        help="Maximum relative HSV saturation change.",
    )
    parser.add_argument(
        "--blur-max-kernel",
        type=int,
        default=5,
        help=(
            "Maximum odd Gaussian blur kernel size. Use 1 to disable "
            "blur."
        ),
    )
    parser.add_argument(
        "--blur-probability",
        type=float,
        default=0.35,
        help="Probability of applying Gaussian blur.",
    )
    parser.add_argument(
        "--noise-max-sigma",
        type=float,
        default=8.0,
        help="Maximum standard deviation of additive Gaussian noise.",
    )
    parser.add_argument(
        "--noise-probability",
        type=float,
        default=0.35,
        help="Probability of adding Gaussian noise.",
    )
    parser.add_argument(
        "--jpeg-quality-min",
        type=int,
        default=55,
        help="Minimum JPEG quality used to simulate compression.",
    )
    parser.add_argument(
        "--jpeg-quality-max",
        type=int,
        default=90,
        help="Maximum JPEG quality used to simulate compression.",
    )
    parser.add_argument(
        "--jpeg-probability",
        type=float,
        default=0.50,
        help="Probability of applying JPEG compression simulation.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Write visualization images with transformed YOLO boxes. "
            "Requires label files."
        ),
    )

    return parser.parse_args()


def list_input_images(
    input_path: Path,
) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"Input image has unsupported extension: {input_path}"
            )

        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    image_paths = [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in: {input_path}"
        )

    return image_paths


def find_label_path(
    image_path: Path,
    labels_folder: Path | None,
) -> Path | None:
    if labels_folder is not None:
        candidate = (
            labels_folder
            / f"{image_path.stem}.txt"
        )

        return candidate if candidate.is_file() else None

    candidate = (
        image_path.parent
        / f"{image_path.stem}.txt"
    )

    return candidate if candidate.is_file() else None


def read_yolo_labels(
    label_path: Path,
) -> list[list[float]]:
    boxes = []

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

        boxes.append(
            [
                int(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
            ]
        )

    return boxes


def yolo_to_xyxy(
    box: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    class_id, x_center, y_center, box_width, box_height = box

    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height

    return x1, y1, x2, y2


def xyxy_to_yolo(
    class_id: int,
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = box

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height

    return [
        class_id,
        x_center,
        y_center,
        box_width,
        box_height,
    ]


def clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(width), x2)
    y2 = min(float(height), y2)

    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None

    return x1, y1, x2, y2


def apply_geometric_transform(
    image: np.ndarray,
    boxes: list[list[float]] | None,
    rng: random.Random,
    args,
) -> tuple[np.ndarray, list[list[float]] | None, dict]:
    height, width = image.shape[:2]

    transform = cv2.getRotationMatrix2D(
        center=(width / 2.0, height / 2.0),
        angle=rng.uniform(
            -args.rotation_max_degrees,
            args.rotation_max_degrees,
        ),
        scale=rng.uniform(
            args.scale_min,
            args.scale_max,
        ),
    )

    translate_x = rng.uniform(
        -args.translation_max_ratio,
        args.translation_max_ratio,
    ) * width
    translate_y = rng.uniform(
        -args.translation_max_ratio,
        args.translation_max_ratio,
    ) * height

    transform[0, 2] += translate_x
    transform[1, 2] += translate_y

    transformed_image = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    transformed_boxes = None

    if boxes is not None:
        transformed_boxes = []

        for box in boxes:
            class_id = int(box[0])
            x1, y1, x2, y2 = yolo_to_xyxy(
                box,
                width,
                height,
            )

            corners = np.array(
                [
                    [x1, y1, 1.0],
                    [x2, y1, 1.0],
                    [x2, y2, 1.0],
                    [x1, y2, 1.0],
                ],
                dtype=np.float64,
            )
            transformed_corners = (
                corners @ transform.T
            )

            transformed_x1 = float(
                transformed_corners[:, 0].min()
            )
            transformed_y1 = float(
                transformed_corners[:, 1].min()
            )
            transformed_x2 = float(
                transformed_corners[:, 0].max()
            )
            transformed_y2 = float(
                transformed_corners[:, 1].max()
            )

            clipped_box = clip_box(
                (
                    transformed_x1,
                    transformed_y1,
                    transformed_x2,
                    transformed_y2,
                ),
                width,
                height,
            )

            if clipped_box is None:
                continue

            transformed_boxes.append(
                xyxy_to_yolo(
                    class_id,
                    clipped_box,
                    width,
                    height,
                )
            )

    # Random crop.
    crop_area_ratio = rng.uniform(
        args.crop_min_area,
        args.crop_max_area,
    )
    crop_width_ratio = np.sqrt(crop_area_ratio)
    crop_height_ratio = crop_area_ratio / crop_width_ratio

    crop_width = max(
        32,
        int(round(width * crop_width_ratio)),
    )
    crop_height = max(
        32,
        int(round(height * crop_height_ratio)),
    )

    crop_x1 = rng.randint(
        0,
        max(0, width - crop_width),
    )
    crop_y1 = rng.randint(
        0,
        max(0, height - crop_height),
    )
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    transformed_image = transformed_image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    if transformed_boxes is not None:
        cropped_boxes = []

        for box in transformed_boxes:
            class_id = int(box[0])
            x1, y1, x2, y2 = yolo_to_xyxy(
                box,
                crop_width,
                crop_height,
            )

            shifted_box = clip_box(
                (
                    x1 - crop_x1,
                    y1 - crop_y1,
                    x2 - crop_x1,
                    y2 - crop_y1,
                ),
                crop_width,
                crop_height,
            )

            if shifted_box is None:
                continue

            cropped_boxes.append(
                xyxy_to_yolo(
                    class_id,
                    shifted_box,
                    crop_width,
                    crop_height,
                )
            )

        transformed_boxes = cropped_boxes

    # Horizontal flip.
    horizontal_flip = (
        not args.no_horizontal_flip
        and rng.random()
        < args.horizontal_flip_probability
    )

    if horizontal_flip:
        transformed_image = cv2.flip(
            transformed_image,
            1,
        )

        if transformed_boxes is not None:
            flipped_boxes = []

            for box in transformed_boxes:
                class_id = int(box[0])
                x1, y1, x2, y2 = yolo_to_xyxy(
                    box,
                    crop_width,
                    crop_height,
                )

                flipped_box = clip_box(
                    (
                        crop_width - x2,
                        y1,
                        crop_width - x1,
                        y2,
                    ),
                    crop_width,
                    crop_height,
                )

                if flipped_box is None:
                    continue

                flipped_boxes.append(
                    xyxy_to_yolo(
                        class_id,
                        flipped_box,
                        crop_width,
                        crop_height,
                    )
                )

            transformed_boxes = flipped_boxes

    # Final resize.
    transformed_image = cv2.resize(
        transformed_image,
        (
            args.output_width,
            args.output_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    if transformed_boxes is not None:
        resized_boxes = []

        for box in transformed_boxes:
            resized_boxes.append(
                xyxy_to_yolo(
                    int(box[0]),
                    yolo_to_xyxy(
                        box,
                        crop_width,
                        crop_height,
                    ),
                    args.output_width,
                    args.output_height,
                )
            )

        transformed_boxes = resized_boxes

    transform_record = {
        "rotation_degrees": float(
            transform[0, 0]
        ),
        "scale": float(
            np.sqrt(
                transform[0, 0] ** 2
                + transform[1, 0] ** 2
            )
        ),
        "translation_x_ratio": float(
            translate_x / width
        ),
        "translation_y_ratio": float(
            translate_y / height
        ),
        "crop": {
            "x1": crop_x1,
            "y1": crop_y1,
            "x2": crop_x2,
            "y2": crop_y2,
            "area_ratio": crop_area_ratio,
        },
        "horizontal_flip": horizontal_flip,
        "output_size": [
            args.output_width,
            args.output_height,
        ],
    }

    return (
        transformed_image,
        transformed_boxes,
        transform_record,
    )


def apply_photometric_transform(
    image: np.ndarray,
    rng: random.Random,
    args,
) -> tuple[np.ndarray, dict]:
    transformed = image.astype(np.float32)

    brightness_change = rng.uniform(
        -args.brightness_max_change,
        args.brightness_max_change,
    )
    contrast_change = rng.uniform(
        -args.contrast_max_change,
        args.contrast_max_change,
    )

    transformed = (
        transformed * (1.0 + contrast_change)
        + 255.0 * brightness_change
    )

    transformed = np.clip(
        transformed,
        0.0,
        255.0,
    ).astype(np.uint8)

    gamma = rng.uniform(
        args.gamma_min,
        args.gamma_max,
    )
    lookup_table = np.array(
        [
            ((index / 255.0) ** gamma) * 255.0
            for index in range(256)
        ],
        dtype=np.uint8,
    )
    transformed = cv2.LUT(
        transformed,
        lookup_table,
    )

    hsv = cv2.cvtColor(
        transformed,
        cv2.COLOR_BGR2HSV,
    ).astype(np.float32)

    hue_change = rng.randint(
        -args.hue_max_change,
        args.hue_max_change,
    )
    saturation_change = rng.uniform(
        -args.saturation_max_change,
        args.saturation_max_change,
    )

    hsv[:, :, 0] = np.clip(
        hsv[:, :, 0] + hue_change,
        0.0,
        179.0,
    )
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1]
        * (1.0 + saturation_change),
        0.0,
        255.0,
    )

    transformed = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    transform_record = {
        "brightness_change": brightness_change,
        "contrast_change": contrast_change,
        "gamma": gamma,
        "hue_change": hue_change,
        "saturation_change": (
            saturation_change
        ),
    }

    return transformed, transform_record


def apply_noise_and_blur(
    image: np.ndarray,
    rng: random.Random,
    args,
) -> tuple[np.ndarray, dict]:
    transform_record = {}

    if (
        args.blur_max_kernel >= 3
        and rng.random() < args.blur_probability
    ):
        kernel_candidates = [
            kernel
            for kernel in range(
                3,
                args.blur_max_kernel + 1,
                2,
            )
        ]

        if kernel_candidates:
            kernel_size = rng.choice(
                kernel_candidates
            )
            image = cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                0,
            )
            transform_record["blur_kernel"] = (
                kernel_size
            )

    if (
        args.noise_max_sigma > 0.0
        and rng.random() < args.noise_probability
    ):
        noise_sigma = rng.uniform(
            1.0,
            args.noise_max_sigma,
        )
        noise = np.random.default_rng(
            rng.randint(0, 2**32 - 1)
        ).normal(
            loc=0.0,
            scale=noise_sigma,
            size=image.shape,
        )

        image = np.clip(
            image.astype(np.float32) + noise,
            0.0,
            255.0,
        ).astype(np.uint8)
        transform_record["noise_sigma"] = (
            noise_sigma
        )

    if (
        args.jpeg_quality_max > 0
        and rng.random() < args.jpeg_probability
    ):
        jpeg_quality = rng.randint(
            args.jpeg_quality_min,
            args.jpeg_quality_max,
        )
        encode_parameters = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            jpeg_quality,
        ]
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            encode_parameters,
        )

        if success:
            image = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR,
            )
            transform_record["jpeg_quality"] = (
                jpeg_quality
            )

    return image, transform_record


def draw_boxes(
    image: np.ndarray,
    boxes: list[list[float]],
) -> np.ndarray:
    visualization = image.copy()
    height, width = visualization.shape[:2]

    for box in boxes:
        class_id = int(box[0])
        x1, y1, x2, y2 = yolo_to_xyxy(
            box,
            width,
            height,
        )

        cv2.rectangle(
            visualization,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            visualization,
            str(class_id),
            (
                int(x1),
                max(20, int(y1) - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return visualization


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.variants_per_image < 0:
        raise ValueError(
            "--variants-per-image must be non-negative"
        )

    if not 0.0 < args.crop_min_area <= args.crop_max_area <= 1.0:
        raise ValueError(
            "Crop area limits must satisfy "
            "0 < crop-min-area <= crop-max-area <= 1"
        )

    if args.scale_min <= 0.0 or args.scale_min > args.scale_max:
        raise ValueError(
            "Scale limits must satisfy 0 < scale-min <= scale-max"
        )

    if args.blur_max_kernel < 1 or args.blur_max_kernel % 2 == 0:
        raise ValueError(
            "--blur-max-kernel must be an odd integer of at least 1"
        )

    if args.output_width < 32 or args.output_height < 32:
        raise ValueError(
            "Output dimensions must be at least 32x32"
        )

    input_images = list_input_images(args.input)

    if args.output.exists():
        shutil.rmtree(args.output)

    (
        args.output / "images"
    ).mkdir(parents=True, exist_ok=True)
    (
        args.output / "labels"
    ).mkdir(parents=True, exist_ok=True)

    if args.visualize:
        (
            args.output / "vis"
        ).mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "input": str(args.input),
        "labels_folder": (
            str(args.labels)
            if args.labels
            else None
        ),
        "variants_per_image": (
            args.variants_per_image
        ),
        "output_size": [
            args.output_width,
            args.output_height,
        ],
        "images": [],
    }
    
    for image_index, image_path in enumerate(
        input_images
    ):
        image = cv2.imread(str(image_path))

        if image is None:
            raise OSError(
                f"Could not read image: {image_path}"
            )

        label_path = find_label_path(
            image_path,
            args.labels,
        )
        original_boxes = (
            read_yolo_labels(label_path)
            if label_path is not None
            else None
        )

        base_name = image_path.stem

        # Preserve the original image.
        original_filename = (
            f"{base_name}_original.jpg"
        )
        original_output_path = (
            args.output
            / "images"
            / original_filename
        )

        if not cv2.imwrite(
            str(original_output_path),
            image,
        ):
            raise OSError(
                "Could not write original image: "
                f"{original_output_path}"
            )

        original_manifest_entry = {
            "source_image": str(image_path),
            "output_image": str(
                original_output_path
            ),
            "label_file": (
                str(label_path)
                if label_path
                else None
            ),
            "variant": "original",
            "transformations": {},
        }

        if original_boxes is not None:
            original_label_path = (
                args.output
                / "labels"
                / f"{base_name}_original.txt"
            )

            original_label_path.write_text(
                "\n".join(
                    " ".join(
                        [
                            str(int(box[0])),
                            f"{box[1]:.6f}",
                            f"{box[2]:.6f}",
                            f"{box[3]:.6f}",
                            f"{box[4]:.6f}",
                        ]
                    )
                    for box in original_boxes
                )
                + "\n",
                encoding="utf-8",
            )

            original_manifest_entry[
                "output_label"
            ] = str(original_label_path)

        manifest["images"].append(
            original_manifest_entry
        )

        # Generate augmented variants.
        for variant_index in range(
            args.variants_per_image
        ):
            variant_seed = random.randint(
                0,
                2**32 - 1,
            )
            variant_rng = random.Random(
                variant_seed
            )

            variant_name = (
                f"{base_name}_variant"
                f"{variant_index:03d}"
            )

            geometric_image, geometric_boxes, geometric_record = (
                apply_geometric_transform(
                    image.copy(),
                    (
                        [box.copy() for box in original_boxes]
                        if original_boxes is not None
                        else None
                    ),
                    variant_rng,
                    args,
                )
            )

            photometric_image, photometric_record = (
                apply_photometric_transform(
                    geometric_image,
                    variant_rng,
                    args,
                )
            )

            final_image, quality_record = (
                apply_noise_and_blur(
                    photometric_image,
                    variant_rng,
                    args,
                )
            )

            output_image_path = (
                args.output
                / "images"
                / f"{variant_name}.jpg"
            )

            if not cv2.imwrite(
                str(output_image_path),
                final_image,
            ):
                raise OSError(
                    "Could not write augmented image: "
                    f"{output_image_path}"
                )

            manifest_entry = {
                "source_image": str(image_path),
                "output_image": str(
                    output_image_path
                ),
                "label_file": (
                    str(label_path)
                    if label_path
                    else None
                ),
                "variant": variant_index,
                "variant_seed": variant_seed,
                "transformations": {
                    **geometric_record,
                    **photometric_record,
                    **quality_record,
                },
            }

            if geometric_boxes is not None:
                output_label_path = (
                    args.output
                    / "labels"
                    / f"{variant_name}.txt"
                )

                output_label_path.write_text(
                    "\n".join(
                        " ".join(
                            [
                                str(int(box[0])),
                                f"{box[1]:.6f}",
                                f"{box[2]:.6f}",
                                f"{box[3]:.6f}",
                                f"{box[4]:.6f}",
                            ]
                        )
                        for box in geometric_boxes
                    )
                    + "\n",
                    encoding="utf-8",
                )

                manifest_entry["output_label"] = str(
                    output_label_path
                )

                if args.visualize:
                    visualization = draw_boxes(
                        final_image,
                        geometric_boxes,
                    )
                    visualization_path = (
                        args.output
                        / "vis"
                        / f"{variant_name}.jpg"
                    )

                    cv2.imwrite(
                        str(visualization_path),
                        visualization,
                    )

            manifest["images"].append(
                manifest_entry
            )

        print(
            f"[{image_index + 1}/"
            f"{len(input_images)}] "
            f"Processed {image_path.name}: "
            f"{args.variants_per_image} variants"
        )

    manifest_path = (
        args.output / "augmentation_manifest.json"
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

    total_images = len(input_images) * (
        args.variants_per_image + 1
    )

    print("\nShared webcam test set created")
    print(f"Input images: {len(input_images)}")
    print(
        f"Variants per image: {args.variants_per_image}"
    )
    print(f"Total output images: {total_images}")
    print(f"Output: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()