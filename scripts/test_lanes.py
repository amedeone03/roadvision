
from pathlib import Path

import cv2
import numpy as np


def fit_lane_line(segments, y_bottom, y_top):
    """
    Fit one straight lane boundary using Hough segments.

    Each segment is (x1, y1, x2, y2).

    Returns:
        (x_bottom, y_bottom, x_top, y_top)
        or None if there are not enough valid points.
    """

    if not segments:
        return None

    x_points = []
    y_points = []
    weights = []

    for x1, y1, x2, y2 in segments:
        length = np.hypot(x2 - x1, y2 - y1)

        if length < 1:
            continue

        # Both endpoints contribute to the fit
        x_points.extend([x1, x2])
        y_points.extend([y1, y2])

        # Longer segments receive more weight
        weights.extend([length, length])

    if len(x_points) < 2:
        return None

    # Weighted linear regression: x = a*y + c
    a, c = np.polyfit(
        y_points,
        x_points,
        deg=1,
        w=np.sqrt(weights)
    )

    # Evaluate the line at two image heights
    x_bottom = int(a * y_bottom + c)
    x_top = int(a * y_top + c)

    return (
        x_bottom,
        y_bottom,
        x_top,
        y_top
    )

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_DIR = Path("runs/lane_experiments")

TARGET_TIME = 5  # Seconds

# Gaussian blur parameters
BLUR_KERNEL = (5, 5)

# Canny thresholds
CANNY_LOW = 50
CANNY_HIGH = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Open the input video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS")


# --------------------------------------------------
# 3. Extract a representative frame
# --------------------------------------------------

target_frame = int(TARGET_TIME * fps)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    target_frame
)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError("Could not read selected frame")

print(f"Extracted frame at approximately {TARGET_TIME} seconds")
print(f"Original shape: {frame.shape}")


# --------------------------------------------------
# 4. Convert to grayscale
# --------------------------------------------------

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)


# --------------------------------------------------
# 5. Apply Gaussian blur
# --------------------------------------------------

blurred = cv2.GaussianBlur(
    gray,
    BLUR_KERNEL,
    0
)


# --------------------------------------------------
# 6. Apply Canny edge detection
# --------------------------------------------------

edges = cv2.Canny(
    blurred,
    CANNY_LOW,
    CANNY_HIGH
)


# --------------------------------------------------
# 7. Define the Region of Interest
# --------------------------------------------------

height, width = edges.shape

# Trapezoidal ROI, defined using relative coordinates
vertices = np.array([
    [
        (int(0.20 * width), height - 1),
        (int(0.43 * width), int(0.52 * height)),
        (int(0.58 * width), int(0.52 * height)),
        (int(0.95 * width), height - 1),
    ]
], dtype=np.int32)


# --------------------------------------------------
# 8. Create ROI mask
# --------------------------------------------------

# Black image with same dimensions as edge map
mask = np.zeros_like(edges)

# Fill the ROI polygon with white
cv2.fillPoly(
    mask,
    vertices,
    255
)


# --------------------------------------------------
# 9. Apply ROI to edge image
# --------------------------------------------------

masked_edges = cv2.bitwise_and(
    edges,
    mask
)

# --------------------------------------------------
# 10. Probabilistic Hough Line Transform
# --------------------------------------------------

lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=40,
    minLineLength=40,
    maxLineGap=30
)

# --------------------------------------------------
# 11. Visualize Hough lines
# --------------------------------------------------

hough_visualization = frame.copy()

line_count = 0

if lines is not None:

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        cv2.line(
            hough_visualization,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=4
        )

        line_count += 1

print(f"Hough line segments detected: {line_count}")

# --------------------------------------------------
# 12. Filter Hough lines by slope
# --------------------------------------------------

filtered_visualization = frame.copy()

left_lines = []
right_lines = []

MIN_ABS_SLOPE = 0.4
MAX_ABS_SLOPE = 3.0

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        dx = x2 - x1
        dy = y2 - y1

        # Avoid division by zero
        if dx == 0:
            continue

        slope = dy / dx

        # Reject horizontal and near-vertical lines
        if not (MIN_ABS_SLOPE <= abs(slope) <= MAX_ABS_SLOPE):
            continue

        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
            color = (0, 255, 0)  # Green
        else:
            right_lines.append((x1, y1, x2, y2))
            color = (255, 0, 0)  # Blue in BGR

        cv2.line(
            filtered_visualization,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            4
        )

print(f"Left candidate segments: {len(left_lines)}")
print(f"Right candidate segments: {len(right_lines)}")

# --------------------------------------------------
# 13. Fit representative lane boundaries
# --------------------------------------------------

height, width = frame.shape[:2]

# Vertical extent of our visualization
y_bottom = int(0.90 * height)
y_top = int(0.68 * height)

left_fit = fit_lane_line(
    left_lines,
    y_bottom,
    y_top
)

right_fit = fit_lane_line(
    right_lines,
    y_bottom,
    y_top
)

# -----------------------------------------
# Validate fitted lane geometry
# -----------------------------------------

if left_fit is not None and right_fit is not None:

    left_bottom_x = left_fit[0]
    left_top_x = left_fit[2]

    right_bottom_x = right_fit[0]
    right_top_x = right_fit[2]

    valid_geometry = (
        left_bottom_x < right_bottom_x
        and left_top_x < right_top_x
    )

    if not valid_geometry:
        print("WARNING: Invalid lane geometry — lines cross!")
    else:
        print("Lane geometry: Valid ordering")

# --------------------------------------------------
# 14. Visualize fitted lane lines
# --------------------------------------------------

fitted_visualization = frame.copy()

if left_fit is not None:
    x1, y1, x2, y2 = left_fit

    cv2.line(
        fitted_visualization,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        8
    )

if right_fit is not None:
    x1, y1, x2, y2 = right_fit

    cv2.line(
        fitted_visualization,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        8
    )

# --------------------------------------------------
# 15. Draw ROI on original frame
# --------------------------------------------------

roi_visualization = frame.copy()

cv2.polylines(
    roi_visualization,
    vertices,
    isClosed=True,
    color=(0, 255, 0),
    thickness=5
)


# --------------------------------------------------
# 16. Save all intermediate results
# --------------------------------------------------

images = {
    "original.png": frame,
    "grayscale.png": gray,
    "blurred.png": blurred,
    "edges.png": edges,
    "roi_mask.png": mask,
    "masked_edges.png": masked_edges,
    "roi_visualization.png": roi_visualization,
    "hough_lines.png": hough_visualization,
    "filtered_lines.png": filtered_visualization,
    "fitted_lanes.png": fitted_visualization,
}

for filename, image in images.items():

    output_path = OUTPUT_DIR / filename

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save {output_path}")

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 17. Print summary
# --------------------------------------------------

print("\n--- LANE PREPROCESSING COMPLETE ---")

print(f"Input resolution: {width}x{height}")
print(f"Grayscale shape: {gray.shape}")
print(f"Blur kernel: {BLUR_KERNEL}")
print(f"Canny thresholds: {CANNY_LOW}, {CANNY_HIGH}")

print("Left fitted boundary:", left_fit)
print("Right fitted boundary:", right_fit)

print("\nAll images saved successfully!")
