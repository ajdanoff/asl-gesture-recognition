# scripts/train_asl_yolo_mp.py
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # or yolo11s.pt for higher accuracy

results = model.train(
    data="datasets/asl_yolo_mp/data.yaml",
    imgsz=416,
    batch=32,
    epochs=100,
    device=0,          # GPU index or 'cpu'
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,          # final learning rate factor (cosine decay end)
    augment=True,
    cos_lr=True,
    close_mosaic=10,
    patience=50,
    project="runs/detect",
    name="asl_yolo11n_mp",
    exist_ok=True,
)