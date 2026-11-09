
from pathlib import Path

import cv2

from src.lanes.lane_detector import detect_lane


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_DIR = Path("runs/lane_inspection")

FRAMES_TO_INSPECT = [
    27, 28, 29, 30,
    65, 66, 67, 68,
    1563, 1564, 1565,
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Open input video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")


# --------------------------------------------------
# 3. Inspect selected frames
# --------------------------------------------------

try:
    for frame_number in FRAMES_TO_INSPECT:

        # CSV frame numbers start at 1.
        # OpenCV frame indices start at 0.
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1,
        )

        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_number}")
            continue

        # Run the existing lane detection algorithm.
        annotated, info = detect_lane(frame)

        # ------------------------------------------
        # Extract fitted boundaries
        # ------------------------------------------

        left_fit = info["left_fit"]
        right_fit = info["right_fit"]

        # ------------------------------------------
        # Create debug visualization
        # ------------------------------------------

        debug_frame = frame.copy()

        # Draw left boundary in green.
        if left_fit is not None:

            x_bottom, y_bottom, x_top, y_top = left_fit

            cv2.line(
                debug_frame,
                (x_bottom, y_bottom),
                (x_top, y_top),
                (0, 255, 0),
                6,
            )

        # Draw right boundary in blue.
        if right_fit is not None:

            x_bottom, y_bottom, x_top, y_top = right_fit

            cv2.line(
                debug_frame,
                (x_bottom, y_bottom),
                (x_top, y_top),
                (255, 0, 0),
                6,
            )

        # ------------------------------------------
        # Display validation status
        # ------------------------------------------

        status = "VALID" if info["valid"] else "REJECTED"

        status_color = (
            (0, 255, 0)
            if info["valid"]
            else (0, 0, 255)
        )

        cv2.putText(
            debug_frame,
            f"Lane geometry: {status}",
            (50, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            status_color,
            3,
        )

        # ------------------------------------------
        # Display candidate segment counts
        # ------------------------------------------

        cv2.putText(
            debug_frame,
            (
                f"Left segments: {info['left_segments']} | "
                f"Right segments: {info['right_segments']}"
            ),
            (50, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # ------------------------------------------
        # Calculate estimated lane widths
        # ------------------------------------------

        if left_fit is not None and right_fit is not None:

            bottom_width = right_fit[0] - left_fit[0]
            top_width = right_fit[2] - left_fit[2]

            cv2.putText(
                debug_frame,
                (
                    f"Bottom width: {bottom_width}px | "
                    f"Top width: {top_width}px"
                ),
                (50, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

        # ------------------------------------------
        # Save images
        # ------------------------------------------

        original_path = (
            OUTPUT_DIR / f"frame_{frame_number}_original.png"
        )

        detected_path = (
            OUTPUT_DIR / f"frame_{frame_number}_detected.png"
        )

        debug_path = (
            OUTPUT_DIR / f"frame_{frame_number}_debug.png"
        )

        cv2.imwrite(str(original_path), frame)
        cv2.imwrite(str(detected_path), annotated)
        cv2.imwrite(str(debug_path), debug_frame)

        # ------------------------------------------
        # Print detection information
        # ------------------------------------------

        print(f"\n--- FRAME {frame_number} ---")
        print(f"Validation: {status}")
        print(f"Left fit: {left_fit}")
        print(f"Right fit: {right_fit}")
        print(f"Left segments: {info['left_segments']}")
        print(f"Right segments: {info['right_segments']}")
        print(f"Debug image: {debug_path}")

finally:
    cap.release()


print("\nLane inspection complete!")
