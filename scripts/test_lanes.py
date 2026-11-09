
from pathlib import Path

import cv2


# -----------------------------------------
# 1. File paths
# -----------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_DIR = Path("runs/lane_experiments")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Open the video
# -----------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video")


# -----------------------------------------
# 3. Extract one frame
# -----------------------------------------

# Select the frame at approximately 5 seconds
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS")

target_time = 5

target_frame = int(target_time * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError("Could not read selected frame")


# -----------------------------------------
# 4. Convert to grayscale
# -----------------------------------------

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)

# -----------------------------------------
# 5. Apply Gaussian blur
# -----------------------------------------

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# -----------------------------------------
# 6. Apply Canny edge detection
# -----------------------------------------

edges = cv2.Canny(
    blurred,
    50,
    150
)

# -----------------------------------------
# 7. Save the results
# -----------------------------------------

cv2.imwrite(
    str(OUTPUT_DIR / "original.png"),
    frame
)

cv2.imwrite(
    str(OUTPUT_DIR / "grayscale.png"),
    gray
)

cv2.imwrite(
    str(OUTPUT_DIR / "blurred.png"),
    blurred
)

cv2.imwrite(
    str(OUTPUT_DIR / "edges.png"),
    edges
)

# -----------------------------------------
# 8. Inspect image dimensions
# -----------------------------------------

print("Original shape:", frame.shape)
print("Grayscale shape:", gray.shape)
print("Edges shape:", edges.shape)

print("\nImages saved successfully!")
