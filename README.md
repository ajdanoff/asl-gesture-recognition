## ASL gesture detection with YOLO

### Dataset

We use the [ASL Alphabet dataset](https://www.kaggle.com/datasets/dorukdemirci/asl-alphabet-dataset) (A–Y, no J/Z).  
Download and unzip it so that the root folder `asl-alphabet-dataset/` contains subfolders like `A-samples/`, `B-samples/`, etc.

### Convert to YOLO format (MediaPipe boxes)

```bash
pip install -r requirements.txt
make prepare-data
```

This creates `datasets/asl_yolo_mp/` with train/val splits and visualizations under `vis/`.

### Train

```bash
make train
```

Weights and metrics will be saved under `runs/detect/asl_yolo11n_mp/`.

### Quick QA: visualize bounding boxes

```bash
make visualize
```

### Inference (webcam)

```bash
make infer
```