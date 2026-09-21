import cv2
import time
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

frame_count = 0
start_time = time.time()

fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # -------------------------
    # FPS
    # -------------------------

    frame_count += 1

    current_time = time.time()
    elapsed = current_time - start_time

    if elapsed >= 1:
        fps = frame_count / elapsed

        frame_count = 0
        start_time = current_time

    fps_text = f"FPS: {fps:.1f}"

    # -------------------------
    # YOLO detection
    # -------------------------

    results = model(frame)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    # -------------------------
    # Display
    # -------------------------

    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("CamOS", frame)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()