# scripts/infer_asl_yolo_mp.py
from ultralytics import YOLO
import cv2

model = YOLO("./models/asl_unified_letters_yolo11n_mp_best.pt")
cap = cv2.VideoCapture(0)

BUFFER_SIZE = 5
gesture_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.6, iou=0.45, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        cls = int(boxes.cls[0])
        conf = float(boxes.conf[0])
        gesture_buffer.append(cls)
        if len(gesture_buffer) > BUFFER_SIZE:
            gesture_buffer.pop(0)
        if len(gesture_buffer) == BUFFER_SIZE and len(set(gesture_buffer)) == 1:
            letter = model.names[cls]
            print(f"Stable gesture: {letter} (conf={conf:.2f})")
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()