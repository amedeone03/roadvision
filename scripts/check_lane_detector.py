from pathlib import Path

import cv2

from src.lanes.lane_detector import detect_lane


video = cv2.VideoCapture("data/videos/test_drive.mp4")

if not video.isOpened():
    raise RuntimeError("Cannot open video")

fps = video.get(cv2.CAP_PROP_FPS)
video.set(cv2.CAP_PROP_POS_FRAMES, int(5 * fps))

success, frame = video.read()
video.release()

if not success:
    raise RuntimeError("Cannot read frame")

annotated, info = detect_lane(frame)

Path("runs/lane_experiments").mkdir(
    parents=True, exist_ok=True
)

cv2.imwrite(
    "runs/lane_experiments/refactored_lane.png",
    annotated
)

print("Lane information:")
print(info)