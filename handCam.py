import cv2
import mediapipe as mp


# -------------------------
# MediaPipe setup
# -------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# -------------------------
# Camera
# -------------------------

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("Failed to open camera.")
    exit()


# -------------------------
# Main loop
# -------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break


    # -------------------------
    # Convert BGR → RGB
    # -------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -------------------------
    # Hand detection
    # -------------------------

    results = hands.process(rgb_frame)


    # -------------------------
    # Draw hand landmarks
    # -------------------------

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # -------------------------
    # Display
    # -------------------------

    cv2.imshow("CamOS - Hands", frame)


    # Press Q to quit
    if cv2.waitKey(1) == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------

cap.release()
hands.close()
cv2.destroyAllWindows()