import cv2
import time

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1
    current_time = time.time()
    elapsed = current_time - start_time

    if elapsed >= 1:
        fps = frame_count / elapsed

        print("FPS:", fps)

        frame_count = 0
        start_time = current_time

    cv2.imshow("CamOS", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()