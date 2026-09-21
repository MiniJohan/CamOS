import cv2
import time
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

frame_count = 0
yolo_frame_count = 0

start_time = time.time()

fps = 0

confidence_threshold = 0.7

last_result = None

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

    yolo_frame_count += 1

    if yolo_frame_count % 2 == 0:
        results = model(frame)
        last_result = results[0]

    # -------------------------
    # Draw detections
    # -------------------------

    if last_result is not None:
        boxes = last_result.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = last_result.boxes.cls.cpu().numpy().astype(int)
        confidences = last_result.boxes.conf.cpu().numpy()

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            if confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = box

            class_name = last_result.names[class_id]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # -------------------------
    # FPS display
    # -------------------------

    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("CamOS", frame)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()