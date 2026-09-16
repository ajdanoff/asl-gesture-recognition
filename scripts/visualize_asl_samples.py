# scripts/visualize_asl_samples.py
import cv2
from pathlib import Path
import random
import os

VIS_ROOT = Path("datasets/asl_yolo_mp/vis/train")
SAMPLES_PER_CLASS = 4
WINDOW_SIZE = (1280, 720)

def main():
    if not VIS_ROOT.exists():
        print("Visualization folder not found:", VIS_ROOT)
        return

    class_dirs = sorted([d for d in VIS_ROOT.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        imgs = list(class_dir.glob("*.jpg"))
        if len(imgs) == 0:
            continue
        chosen = random.sample(imgs, min(SAMPLES_PER_CLASS, len(imgs)))

        for i, img_path in enumerate(chosen):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(WINDOW_SIZE[0] / w, WINDOW_SIZE[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            win_name = f"{class_name} - {img_path.name}"
            cv2.imshow(win_name, img_resized)

        print(f"Showing {len(chosen)} samples for class {class_name}. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Done.")

if __name__ == "__main__":
    main()