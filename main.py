import cv2
import time
from ultralytics import YOLO


# -------------------------
# Model
# -------------------------

model = YOLO("yolo11n.pt")

confidence_threshold = 0.7
inference_size = 416

# Run YOLO every Nth frame
frame_skip = 2


# -------------------------
# Camera
# -------------------------

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("Failed to open camera.")
    exit()


# -------------------------
# Performance tracking
# -------------------------

frame_count = 0
yolo_frame_count = 0

fps = 0
inference_time_ms = 0

fps_start_time = time.perf_counter()

last_result = None


# -------------------------
# Main loop
# -------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break


    # -------------------------
    # Camera FPS
    # -------------------------

    frame_count += 1

    current_time = time.perf_counter()
    elapsed = current_time - fps_start_time

    if elapsed >= 1.0:

        fps = frame_count / elapsed

        frame_count = 0
        fps_start_time = current_time


    # -------------------------
    # YOLO detection
    # -------------------------

    yolo_frame_count += 1

    if yolo_frame_count % frame_skip == 0:

        inference_start = time.perf_counter()

        results = model(
            frame,
            imgsz = inference_size,
            conf = confidence_threshold,
            verbose = False
        )

        inference_end = time.perf_counter()

        inference_time_ms = (
            inference_end - inference_start
        ) * 1000

        last_result = results[0]


    # -------------------------
    # Draw detections
    # -------------------------

    if last_result is not None:

        boxes = (
            last_result.boxes.xyxy
            .cpu()
            .numpy()
            .astype(int)
        )

        class_ids = (
            last_result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            last_result.boxes.conf
            .cpu()
            .numpy()
        )


        for box, class_id, confidence in zip(
            boxes,
            class_ids,
            confidences
        ):

            x1, y1, x2, y2 = box

            class_name = last_result.names[class_id]

            label = f"{class_name} {confidence:.2f}"


            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


            # Label
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
    # Performance display
    # -------------------------

    fps_text = f"FPS: {fps:.1f}"

    inference_text = (
        f"YOLO: {inference_time_ms:.1f} ms"
    )


    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )


    cv2.putText(
        frame,
        inference_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )


    # -------------------------
    # Display
    # -------------------------

    cv2.imshow("CamOS", frame)


    # Press Q to quit
    if cv2.waitKey(1) == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------

cap.release()
cv2.destroyAllWindows()