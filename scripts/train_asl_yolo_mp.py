from ultralytics import YOLO


model = YOLO("yolo11n.pt")

model.train(
    data="/content/unified_datasets/asl_unified_digits_yolo_mp/data.yaml",
    imgsz=512,
    batch=32,
    epochs=100,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    close_mosaic=10,
    patience=30,
    seed=42,
    deterministic=True,
    project="runs/detect",
    name="asl_unified_digits_yolo11n_mp_v1",
    exist_ok=False,
)

from ultralytics import YOLO

model = YOLO("runs/detect/runs/detect/asl_unified_digits_yolo11n_mp_v1/weights/best.pt")

metrics = model.val(
    data="/content/unified_datasets/asl_unified_digits_yolo_mp/data.yaml",
    imgsz=512,
    conf=0.001,
    iou=0.45,
    device=0,
    plots=True,
    save_json=True,
)

print(metrics)

from ultralytics import YOLO


# Train a letters-only YOLO11n detector on the converted dataset.
model = YOLO("yolo11n.pt")

model.train(
    data="/content/unified_datasets/asl_unified_letters_yolo_mp/data.yaml",
    imgsz=512,
    batch=32,
    epochs=100,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    close_mosaic=10,
    patience=30,
    seed=42,
    deterministic=True,
    project="runs/detect",
    name="asl_unified_letters_yolo11n_mp_v1",
    exist_ok=False,
)

from ultralytics import YOLO


model = YOLO(
    "runs/detect/runs/detect/asl_unified_letters_yolo11n_mp_v1/weights/best.pt"
)

metrics = model.val(
    data="/content/unified_datasets/asl_unified_letters_yolo_mp/data.yaml",
    imgsz=512,
    batch=32,
    conf=0.001,
    iou=0.45,
    device=0,
    plots=True,
    save_json=True,
)

print(metrics)