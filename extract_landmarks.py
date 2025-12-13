import cv2
import os
import numpy as np
import mediapipe as mp


DATA_DIR = r"C:\Users\vedaa\data\raw_videos"
OUT_DIR  = r"C:\Users\vedaa\data\landmarks"
TARGET_FRAMES = 64

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs(OUT_DIR, exist_ok=True)


def extract_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame_rgb)

        frame_landmarks = np.zeros((2, 21, 3))  # 2 hands

        if res.multi_hand_landmarks:
            for i, hand in enumerate(res.multi_hand_landmarks[:2]):
                for j, lm in enumerate(hand.landmark):
                    frame_landmarks[i, j] = [lm.x, lm.y, lm.z]

        frames.append(frame_landmarks)

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames)

    # pad / trim to TARGET_FRAMES
    if len(frames) < TARGET_FRAMES:
        pad = np.zeros((TARGET_FRAMES - len(frames), 2, 21, 3))
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:TARGET_FRAMES]

    return frames.reshape(TARGET_FRAMES, -1)  # (64, 126)


for sign in os.listdir(DATA_DIR):
    sign_path = os.path.join(DATA_DIR, sign)
    if not os.path.isdir(sign_path):
        continue

    out_sign_dir = os.path.join(OUT_DIR, sign)
    os.makedirs(out_sign_dir, exist_ok=True)

    for vid in os.listdir(sign_path):
        if not vid.endswith(".mp4"):
            continue

        vid_path = os.path.join(sign_path, vid)
        print("Processing:", vid_path)

        data = extract_from_video(vid_path)
        if data is None:
            print(" Failed:", vid_path)
            continue

        out_file = os.path.join(
            out_sign_dir,
            vid.replace(".mp4", ".npy")
        )
        np.save(out_file, data)

print(" LANDMARK EXTRACTION COMPLETE")
