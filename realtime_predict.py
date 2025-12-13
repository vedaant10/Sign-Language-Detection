
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import time


CAMERA_INDEX = 1          
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

SEQUENCE_LEN = 64
CONFIDENCE_THRESHOLD = 0.6
MOTION_THRESHOLD = 0.015

MODEL_PATH = r"C:\Users\vedaa\models\sign_model.h5"
LANDMARK_DIR = r"C:\Users\vedaa\data\landmarks"


labels = sorted(os.listdir(LANDMARK_DIR))
model = load_model(MODEL_PATH)


HINDI_MAP = {
    "hi": "नमस्ते",
    "bye": "अलविदा",
    "yes": "हाँ",
    "no": "नहीं",
    "thank_you": "धन्यवाद",
    "help": "मदद",
    "hungry": "भूख लगी है",
}


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(1.5)


sequence = []
prev_landmarks = None
state = "IDLE"
display_text = ""


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # FORCE SAME RESOLUTION AS TRAINING
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks = np.zeros((2, 21, 3))
    if results.multi_hand_landmarks:
        for i, hand in enumerate(results.multi_hand_landmarks[:2]):
            for j, lm in enumerate(hand.landmark):
                landmarks[i, j] = [lm.x, lm.y, lm.z]

    flat = landmarks.reshape(-1)


    motion = 0.0
    if prev_landmarks is not None:
        motion = np.mean(np.abs(flat - prev_landmarks))
    prev_landmarks = flat


    if state == "IDLE" and motion > MOTION_THRESHOLD:
        sequence = []
        display_text = ""
        state = "RECORDING"

    if state == "RECORDING":
        sequence.append(flat)
        sequence = sequence[-SEQUENCE_LEN:]

        if motion < MOTION_THRESHOLD and len(sequence) == SEQUENCE_LEN:
            x = np.expand_dims(sequence, axis=0)
            preds = model.predict(x, verbose=0)[0]
            confidence = np.max(preds)
            label = labels[np.argmax(preds)]

            if confidence > CONFIDENCE_THRESHOLD and label != "idle":
                display_text = HINDI_MAP.get(label, label)

            state = "PREDICTED"

    if state == "PREDICTED" and motion < MOTION_THRESHOLD * 0.3:
        state = "IDLE"


    cv2.putText(
        frame,
        f"Prediction: {display_text}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"State: {state}",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.imshow("SIGN LANGUAGE RECOGNITION", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()


