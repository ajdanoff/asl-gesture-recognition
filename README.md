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

## Benchmark datasets

Two held-out webcam benchmark datasets are available:

- [ASL Webcam Benchmark — Original](https://huggingface.co/datasets/ajdanoff/asl-webcam-benchmark-original)
- [ASL Webcam Benchmark — Augmented](https://huggingface.co/datasets/ajdanoff/asl-webcam-benchmark-augmented)

The original benchmark contains one image per gesture class. The augmented
benchmark contains the original image plus five controlled augmented variants
per class.

Both datasets use a shared 36-class YOLO label space:

- `0–9`: digits
- `10–35`: letters `A–Z`

Models are evaluated using symbol-based matching, which correctly supports the
unified model, digit-only model, and letter-only model despite their different
internal class-index ranges.

See the [Benchmark datasets wiki page](https://github.com/ajdanoff/gesture-swarm-ros2/wiki/Benchmark-datasets)
for the evaluation protocol, current results, and interpretation.
