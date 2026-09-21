import cv2

cap = cv2.VideoCapture(0)

print(cap.isOpened())

ret, frame = cap.read()

print(ret)
print(frame.shape)