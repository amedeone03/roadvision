
from pathlib import Path

import cv2

from src.lanes.lane_detector import detect_lane


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_DIR = Path("runs/lane_experiments")

TARGET_TIME = 5.0

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. Read the target frame
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS")

target_frame = int(TARGET_TIME * fps)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    target_frame,
)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError(
        f"Could not read frame {target_frame}"
    )


# --------------------------------------------------
# 3. Run both fitting methods
# --------------------------------------------------

annotated, info, debug_frame = detect_lane(
    frame,
    return_debug=True,
)


# --------------------------------------------------
# 4. Compare estimated lane widths
# --------------------------------------------------

def print_fit_summary(name, left_fit, right_fit, valid):

    print(f"\n--- {name} ---")

    print(f"Left fit: {left_fit}")
    print(f"Right fit: {right_fit}")
    print(f"Geometry valid: {valid}")

    if left_fit is None or right_fit is None:
        print("Lane width unavailable")
        return

    bottom_width = right_fit[0] - left_fit[0]
    top_width = right_fit[2] - left_fit[2]

    print(f"Bottom width: {bottom_width} px")
    print(f"Top width: {top_width} px")


print("\n--- ROADVISION FITTING COMPARISON ---")

print(f"Frame index: {target_frame}")
print(f"Timestamp: {target_frame / fps:.2f} s")

print(f"Left candidate segments: {info['left_segments']}")
print(f"Right candidate segments: {info['right_segments']}")

print_fit_summary(
    "WEIGHTED LEAST SQUARES",
    info["left_fit"],
    info["right_fit"],
    info["valid"],
)

print_fit_summary(
    "RANSAC",
    info["left_ransac"],
    info["right_ransac"],
    info["ransac_valid"],
)


# --------------------------------------------------
# 5. Save diagnostic images
# --------------------------------------------------

images = {
    OUTPUT_DIR / "ransac_original.png": frame,
    OUTPUT_DIR / "ransac_weighted_output.png": annotated,
    OUTPUT_DIR / "ransac_comparison.png": debug_frame,
}

for path, image in images.items():

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(
            f"Could not save image: {path}"
        )


print("\n--- SAVED IMAGES ---")

for path in images:
    print(path)

print("\nRANSAC comparison complete!")
