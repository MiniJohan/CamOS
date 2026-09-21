import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

ret, frame = cap.read()

if ret:
    results = model(frame)

    print("YOLO finished!")
    print(results)

cap.release()
cv2.destroyAllWindows()