import cv2
import time
import os


CAMERA_INDEX = 1        
FPS = 30
RECORD_SECONDS = 5
CLIPS_PER_SIGN = 15

SIGNS = ["Idle"]


ROOT = r"C:\Users\vedaa\data\raw_videos"


os.makedirs(ROOT, exist_ok=True)
for s in SIGNS:
    os.makedirs(os.path.join(ROOT, s), exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(1.5)

if not cap.isOpened():
    print(" Camera failed to open")
    exit()

print(" Camera opened")
print("Press Q at any time to quit")


for sign in SIGNS:
    print(f"\n=== SIGN: {sign} ===")

    for clip in range(CLIPS_PER_SIGN):
        print(f"Get ready for clip {clip+1}/{CLIPS_PER_SIGN}")
        time.sleep(2)

        filename = os.path.join(
            ROOT, sign, f"{sign}_{clip}_{int(time.time())}.mp4"
        )

        # Grab ONE frame to get real size
        ret, frame = cap.read()
        if not ret:
            print(" Failed to read frame")
            continue

        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, FPS, (w, h))

        start = time.time()
        while time.time() - start < RECORD_SECONDS:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.putText(
                frame,
                f"{sign} | recording...",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.imshow("RECORD", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                exit()

        out.release()
        print(f" Saved: {filename}")

print("\n ALL RECORDINGS DONE")
cap.release()
cv2.destroyAllWindows()
