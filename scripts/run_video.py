from pathlib import Path
import time

import cv2
from ultralytics import YOLO


# 1. Configuration
VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_PATH = Path("runs/detection_output.mp4")
MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.35
ROAD_USER_CLASSES = [0, 1, 2, 3, 5, 7]


# 2. Load YOLO
model = YOLO(MODEL_PATH)


# 3. Open input video
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    raise RuntimeError("Invalid video FPS")

print(f"Video: {width}x{height} at {fps:.1f} FPS")


# 4. Prepare output
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video")


# 5. Process every frame
frame_count = 0
total_time = 0.0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        start = time.perf_counter()

        results = model.predict(
            frame,
            conf=CONFIDENCE,
            classes=ROAD_USER_CLASSES,
            device="mps",
            verbose=False,
        )

        annotated = results[0].plot()

        elapsed = time.perf_counter() - start
        total_time += elapsed
        frame_count += 1

        writer.write(annotated)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

finally:
    cap.release()
    writer.release()


# 6. Report performance
if frame_count > 0:
    avg_time = total_time / frame_count

    print("\nProcessing complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Average processing time: {avg_time * 1000:.1f} ms")
    print(f"Processing FPS: {1 / avg_time:.1f}")
    print(f"Output saved to: {OUTPUT_PATH}")
