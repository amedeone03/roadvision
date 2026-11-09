
from pathlib import Path

import cv2
import numpy as np


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
# 10. Draw ROI on original frame
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
# 11. Save all intermediate results
# --------------------------------------------------

images = {
    "original.png": frame,
    "grayscale.png": gray,
    "blurred.png": blurred,
    "edges.png": edges,
    "roi_mask.png": mask,
    "masked_edges.png": masked_edges,
    "roi_visualization.png": roi_visualization,
}

for filename, image in images.items():

    output_path = OUTPUT_DIR / filename

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save {output_path}")

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 12. Print summary
# --------------------------------------------------

print("\n--- LANE PREPROCESSING COMPLETE ---")

print(f"Input resolution: {width}x{height}")
print(f"Grayscale shape: {gray.shape}")
print(f"Blur kernel: {BLUR_KERNEL}")
print(f"Canny thresholds: {CANNY_LOW}, {CANNY_HIGH}")

print("\nAll images saved successfully!")
