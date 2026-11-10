

from pathlib import Path

import cv2
import numpy as np


VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_DIR = Path("runs/lane_experiments")

TARGET_TIME = 5.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_filtered_segments(edges, vertices, width):
    """Apply an ROI and return slope-filtered Hough segments."""

    mask = np.zeros_like(edges)

    cv2.fillPoly(mask, vertices, 255)

    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=30,
    )

    left_segments = []
    right_segments = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            dx = x2 - x1

            if dx == 0:
                continue

            slope = (y2 - y1) / dx
            mid_x = (x1 + x2) / 2

            if not 0.4 <= abs(slope) <= 3.0:
                continue

            if slope < 0 and mid_x < width / 2:
                left_segments.append((x1, y1, x2, y2))

            elif slope > 0 and mid_x > width / 2:
                right_segments.append((x1, y1, x2, y2))

    return left_segments, right_segments


def draw_segments(frame, vertices, left, right):
    """Visualize selected lane candidates."""

    image = frame.copy()

    cv2.polylines(
        image,
        vertices,
        True,
        (0, 255, 255),
        2,
    )

    for x1, y1, x2, y2 in left:
        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4,
        )

    for x1, y1, x2, y2 in right:
        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            4,
        )

    return image


# ------------------------------------------
# Read the test frame
# ------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid FPS")

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    int(TARGET_TIME * fps),
)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Cannot read target frame")

height, width = frame.shape[:2]


# ------------------------------------------
# Preprocessing
# ------------------------------------------

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY,
)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0,
)

edges = cv2.Canny(
    blurred,
    50,
    150,
)


# ------------------------------------------
# Define both masks
# ------------------------------------------

expanded_roi = np.array([[
    (int(0.20 * width), height - 1),
    (int(0.38 * width), int(0.48 * height)),
    (int(0.62 * width), int(0.48 * height)),
    (int(0.95 * width), height - 1),
]], dtype=np.int32)

road_roi = np.array([[
    (int(0.20 * width), height - 1),
    (int(0.41 * width), int(0.60 * height)),
    (int(0.59 * width), int(0.60 * height)),
    (int(0.95 * width), height - 1),
]], dtype=np.int32)


# ------------------------------------------
# Compare both masks
# ------------------------------------------

for name, vertices in [
    ("expanded", expanded_roi),
    ("road_focused", road_roi),
]:
    left, right = get_filtered_segments(
        edges,
        vertices,
        width,
    )

    image = draw_segments(
        frame,
        vertices,
        left,
        right,
    )

    output = OUTPUT_DIR / f"{name}_candidates.png"

    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Cannot save {output}")

    print(f"\n{name}")
    print(f"Left segments: {len(left)}")
    print(f"Right segments: {len(right)}")
    print(f"Saved: {output}")
