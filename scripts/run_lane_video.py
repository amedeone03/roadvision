
from pathlib import Path
import time

import cv2

from src.lanes.lane_detector import detect_lane


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_PATH = Path("runs/lane_output.mp4")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Open input video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video metadata")

print("\n--- ROADVISION LANE DETECTION ---")
print(f"Input video: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"Input FPS: {fps:.1f}")


# --------------------------------------------------
# 3. Create output video
# --------------------------------------------------

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video: {OUTPUT_PATH}"
    )


# --------------------------------------------------
# 4. Initialize statistics
# --------------------------------------------------

frame_count = 0

valid_lane_frames = 0
invalid_lane_frames = 0

total_lane_processing_time = 0.0

start_total = time.perf_counter()


# --------------------------------------------------
# 5. Process video frame by frame
# --------------------------------------------------

try:
    while True:

        # Read next frame
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # --------------------------------------
        # Run lane detection
        # --------------------------------------

        start_detection = time.perf_counter()

        annotated_frame, lane_info = detect_lane(frame)

        detection_time = (
            time.perf_counter() - start_detection
        )

        total_lane_processing_time += detection_time

        # --------------------------------------
        # Evaluate geometric validity
        # --------------------------------------

        if lane_info["valid"]:
            valid_lane_frames += 1
        else:
            invalid_lane_frames += 1

        # --------------------------------------
        # Save annotated frame
        # --------------------------------------

        writer.write(annotated_frame)

        # --------------------------------------
        # Print progress
        # --------------------------------------

        if frame_count % 100 == 0:

            print(
                f"Processed {frame_count} frames | "
                f"Valid: {valid_lane_frames} | "
                f"Invalid: {invalid_lane_frames}"
            )

finally:

    # Always release video resources
    cap.release()
    writer.release()


# --------------------------------------------------
# 6. Calculate performance
# --------------------------------------------------

total_runtime = time.perf_counter() - start_total

print("\n--- LANE DETECTION RESULTS ---")

print(f"Frames processed: {frame_count}")
print(f"Valid lane frames: {valid_lane_frames}")
print(f"Invalid lane frames: {invalid_lane_frames}")

if frame_count > 0:

    validation_rate = valid_lane_frames / frame_count

    avg_detection_time = (
        total_lane_processing_time / frame_count
    )

    detection_fps = (
        1 / avg_detection_time
        if avg_detection_time > 0
        else 0
    )

    end_to_end_fps = (
        frame_count / total_runtime
        if total_runtime > 0
        else 0
    )

    print(
        f"Lane validation rate: "
        f"{validation_rate:.2%}"
    )

    print(
        f"Average lane processing time: "
        f"{avg_detection_time * 1000:.2f} ms/frame"
    )

    print(
        f"Lane processing FPS: "
        f"{detection_fps:.2f}"
    )

    print(
        f"End-to-end throughput: "
        f"{end_to_end_fps:.2f} FPS"
    )

print(f"Output video: {OUTPUT_PATH}")
print("\nProcessing complete!")
