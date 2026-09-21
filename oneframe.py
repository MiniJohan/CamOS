import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

ret, frame = cap.read()

if ret:
    results = model(frame)

    result = results[0]

    print("YOLO finished!")
    print("Objects detected:", len(result.boxes))
    print("Bounding boxes:", result.boxes.xyxy)
    print("Confidence:", result.boxes.conf)
    print("Class IDs:", result.boxes.cls)

cap.release()
cv2.destroyAllWindows()