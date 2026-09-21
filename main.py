import cv2

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print(frame.shape)
    print(frame.dtype)
    break

    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("CamOS", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()