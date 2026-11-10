
from pathlib import Path

import cv2
import numpy as np

from src.lanes.lane_detector import detect_lane


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")

OUTPUT_PATH = Path(
    "runs/lane_experiments/perspective_lane.png"
)

TARGET_TIME = 5.0

# Manually selected experimental vanishing point.
# These coordinates are NOT camera-calibrated.
VP_X = 780
VP_Y = 560

# Top of the association polygon
POLYGON_TOP_Y = 600

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. Construct perspective-constrained boundary
# --------------------------------------------------

def interpolate_boundary(
    bottom_x,
    bottom_y,
    vp_x,
    vp_y,
    target_y,
):
    """
    Find X at target_y on the line connecting
    a bottom anchor to the vanishing point.
    """

    if bottom_y == vp_y:
        raise ValueError("Invalid vanishing point height")

    fraction = (
        (target_y - vp_y)
        / (bottom_y - vp_y)
    )

    return (
        vp_x + fraction * (bottom_x - vp_x)
    )


# --------------------------------------------------
# 3. Open video and read frame
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid input FPS")

target_frame = int(TARGET_TIME * fps)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    target_frame,
)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError("Could not read target frame")


# --------------------------------------------------
# 4. Obtain near-field lane estimates
# --------------------------------------------------

_, lane_info = detect_lane(frame)

left_fit = lane_info["left_fit"]
right_fit = lane_info["right_fit"]

if left_fit is None or right_fit is None:
    raise RuntimeError(
        "Both near-field boundaries are required"
    )

left_bottom_x, bottom_y = left_fit[:2]
right_bottom_x, right_bottom_y = right_fit[:2]

if bottom_y != right_bottom_y:
    raise RuntimeError("Lane anchors have different heights")

if not (0 <= VP_Y < POLYGON_TOP_Y < bottom_y):
    raise ValueError("Invalid vertical geometry")

if left_bottom_x >= right_bottom_x:
    raise ValueError("Bottom lane anchors are reversed")


# --------------------------------------------------
# 5. Construct constrained lane boundaries
# --------------------------------------------------

left_top_x = interpolate_boundary(
    left_bottom_x,
    bottom_y,
    VP_X,
    VP_Y,
    POLYGON_TOP_Y,
)

right_top_x = interpolate_boundary(
    right_bottom_x,
    bottom_y,
    VP_X,
    VP_Y,
    POLYGON_TOP_Y,
)

polygon = np.array(
    [
        [left_bottom_x, bottom_y],
        [left_top_x, POLYGON_TOP_Y],
        [right_top_x, POLYGON_TOP_Y],
        [right_bottom_x, bottom_y],
    ],
    dtype=np.float32,
)


# --------------------------------------------------
# 6. Visualization
# --------------------------------------------------

debug = frame.copy()
overlay = frame.copy()

# Fill the perspective-constrained polygon
cv2.fillPoly(
    overlay,
    [np.rint(polygon).astype(np.int32)],
    (0, 255, 0),
)

debug = cv2.addWeighted(
    overlay,
    0.20,
    debug,
    0.80,
    0,
)

# Left boundary in green
cv2.line(
    debug,
    (int(left_bottom_x), int(bottom_y)),
    (round(left_top_x), POLYGON_TOP_Y),
    (0, 255, 0),
    5,
)

# Right boundary in blue
cv2.line(
    debug,
    (int(right_bottom_x), int(bottom_y)),
    (round(right_top_x), POLYGON_TOP_Y),
    (255, 0, 0),
    5,
)

# Manual vanishing point in red
cv2.circle(
    debug,
    (VP_X, VP_Y),
    9,
    (0, 0, 255),
    -1,
)

cv2.putText(
    debug,
    "Manual VP",
    (VP_X + 15, VP_Y - 15),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2,
)


# --------------------------------------------------
# 7. Save and report
# --------------------------------------------------

if not cv2.imwrite(str(OUTPUT_PATH), debug):
    raise RuntimeError("Could not save output image")

print("\n--- PERSPECTIVE LANE EXPERIMENT ---")

print(f"Frame index: {target_frame}")
print(f"Existing geometry valid: {lane_info['valid']}")

print(f"Bottom-left anchor: {left_bottom_x}")
print(f"Bottom-right anchor: {right_bottom_x}")

print(f"Manual vanishing point: ({VP_X}, {VP_Y})")
print(f"Polygon top Y: {POLYGON_TOP_Y}")

print(f"Constrained left top X: {left_top_x:.1f}")
print(f"Constrained right top X: {right_top_x:.1f}")

print(
    f"Top lane width: "
    f"{right_top_x - left_top_x:.1f} px"
)

print(f"Output: {OUTPUT_PATH}")
