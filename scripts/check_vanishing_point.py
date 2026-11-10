
from pathlib import Path

import cv2

from src.lanes.lane_detector import detect_lane
from src.lanes.perspective import estimate_vanishing_point


VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_PATH = Path("runs/lane_experiments/vanishing_point.png")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(5 * fps))

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Cannot read target frame")

_, lane_info = detect_lane(frame)

left_fit = lane_info["left_fit"]
right_fit = lane_info["right_fit"]

vp = estimate_vanishing_point(left_fit, right_fit)

debug = frame.copy()

# Draw fitted boundaries even if geometry is rejected.
for fit, color in [
    (left_fit, (0, 255, 0)),
    (right_fit, (255, 0, 0)),
]:
    if fit is not None:
        cv2.line(
            debug,
            (fit[0], fit[1]),
            (fit[2], fit[3]),
            color,
            4,
        )

if vp is not None:
    x, y = vp

    print(f"Vanishing point: x={x:.1f}, y={y:.1f}")

    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
        center = (round(x), round(y))

        cv2.circle(debug, center, 12, (0, 0, 255), -1)
        cv2.putText(
            debug,
            "Estimated VP",
            (center[0] + 20, center[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
else:
    print("Vanishing point unavailable")

if not cv2.imwrite(str(OUTPUT_PATH), debug):
    raise RuntimeError("Could not save diagnostic image")

print(f"Saved: {OUTPUT_PATH}")
