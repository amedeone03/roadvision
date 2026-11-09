
from pathlib import Path
import csv
import time

import cv2
import numpy as np

from src.lanes.lane_detector import detect_lane
from src.lanes.temporal_filter import TemporalLaneFilter
from src.lanes.lane_smoother import LaneSmoother


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")

OUTPUT_PATH = Path("runs/lane_smoothed_output.mp4")
CSV_PATH = Path("runs/lane_smoothed_results.csv")

# Experimental temporal filter parameters
MAX_JUMP = 100
MAX_REFERENCE_AGE = 5
MAX_CANDIDATE_GAP = 2

# Exponential moving average parameter
EMA_ALPHA = 0.3

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Draw smoothed lane
# --------------------------------------------------

def draw_smoothed_lane(frame, left_fit, right_fit):
    """
    Draw the smoothed ego-lane polygon.

    The input frame is never modified in place.
    """

    annotated = frame.copy()

    if left_fit is None or right_fit is None:
        return annotated

    lb_x, lb_y, lt_x, lt_y = left_fit
    rb_x, rb_y, rt_x, rt_y = right_fit

    polygon = np.array([[
        [lb_x, lb_y],
        [lt_x, lt_y],
        [rt_x, rt_y],
        [rb_x, rb_y],
    ]], dtype=np.int32)

    overlay = frame.copy()

    # Semi-transparent green lane area
    cv2.fillPoly(
        overlay,
        polygon,
        (0, 255, 0),
    )

    annotated = cv2.addWeighted(
        overlay,
        0.25,
        frame,
        0.75,
        0,
    )

    # Left boundary: green
    cv2.line(
        annotated,
        (lb_x, lb_y),
        (lt_x, lt_y),
        (0, 255, 0),
        6,
    )

    # Right boundary: blue
    cv2.line(
        annotated,
        (rb_x, rb_y),
        (rt_x, rt_y),
        (255, 0, 0),
        6,
    )

    return annotated


# --------------------------------------------------
# 3. Open input video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open input video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video metadata")

print("\n--- ROADVISION: SMOOTHED LANE DETECTION ---")
print(f"Input video: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"Input FPS: {fps:.1f}")

print(f"Maximum jump: {MAX_JUMP} px")
print(f"Maximum reference age: {MAX_REFERENCE_AGE} frames")
print(f"Maximum candidate gap: {MAX_CANDIDATE_GAP} frames")
print(f"EMA alpha: {EMA_ALPHA}")


# --------------------------------------------------
# 4. Create output video
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
# 5. Initialize stateful components
# --------------------------------------------------

lane_filter = TemporalLaneFilter(
    max_jump=MAX_JUMP,
    max_reference_age=MAX_REFERENCE_AGE,
    max_candidate_gap=MAX_CANDIDATE_GAP,
)

lane_smoother = LaneSmoother(
    alpha=EMA_ALPHA,
)

frame_count = 0

geometrically_valid_frames = 0
geometrically_invalid_frames = 0

temporally_accepted_frames = 0
temporally_rejected_frames = 0

total_lane_processing_time = 0.0

start_total = time.perf_counter()


# --------------------------------------------------
# 6. Process video and write CSV
# --------------------------------------------------

try:
    with CSV_PATH.open("w", newline="") as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "frame",
            "timestamp",
            "valid",
            "left_segments",
            "right_segments",
            "left_bottom_x",
            "right_bottom_x",
            "left_top_x",
            "right_top_x",
            "bottom_width",
            "top_width",
            "temporal_accepted",
            "temporal_reason",
            "smoothed_left_bottom_x",
            "smoothed_right_bottom_x",
            "smoothed_left_top_x",
            "smoothed_right_top_x",
        ])

        while True:

            # --------------------------------------
            # Read next video frame
            # --------------------------------------

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # First video frame corresponds to t = 0.
            timestamp = (frame_count - 1) / fps

            # --------------------------------------
            # 6.1 Raw lane detection
            # --------------------------------------

            start_detection = time.perf_counter()

            _, lane_info = detect_lane(frame)

            detection_time = (
                time.perf_counter() - start_detection
            )

            total_lane_processing_time += detection_time

            # We discard the image returned by
            # detect_lane(), because it contains the
            # raw polygon. We will draw the smoothed
            # polygon after temporal filtering.

            # --------------------------------------
            # 6.2 Geometric validation
            # --------------------------------------

            valid = lane_info["valid"]

            left_fit = lane_info["left_fit"]
            right_fit = lane_info["right_fit"]

            left_segments = lane_info["left_segments"]
            right_segments = lane_info["right_segments"]

            if valid:
                geometrically_valid_frames += 1
            else:
                geometrically_invalid_frames += 1

            # --------------------------------------
            # 6.3 Temporal consistency filtering
            # --------------------------------------

            temporal_accepted, temporal_reason = (
                lane_filter.update(
                    frame_count,
                    lane_info,
                )
            )

            if temporal_accepted:
                temporally_accepted_frames += 1
            else:
                temporally_rejected_frames += 1

            # --------------------------------------
            # 6.4 EMA smoothing
            # --------------------------------------

            smoothed_left, smoothed_right = (
                lane_smoother.update(
                    lane_info,
                    temporal_accepted,
                    temporal_reason,
                )
            )

            # --------------------------------------
            # 6.5 Draw final polygon
            # --------------------------------------

            if temporal_accepted:

                annotated_frame = draw_smoothed_lane(
                    frame,
                    smoothed_left,
                    smoothed_right,
                )

            else:

                # Never display an unaccepted polygon.
                annotated_frame = frame.copy()

            # --------------------------------------
            # 6.6 Extract raw coordinates
            # --------------------------------------

            left_bottom_x = None
            right_bottom_x = None

            left_top_x = None
            right_top_x = None

            bottom_width = None
            top_width = None

            if left_fit is not None:
                left_bottom_x = left_fit[0]
                left_top_x = left_fit[2]

            if right_fit is not None:
                right_bottom_x = right_fit[0]
                right_top_x = right_fit[2]

            if left_fit is not None and right_fit is not None:

                bottom_width = (
                    right_bottom_x - left_bottom_x
                )

                top_width = (
                    right_top_x - left_top_x
                )

            # --------------------------------------
            # 6.7 Extract smoothed coordinates
            # --------------------------------------

            smoothed_left_bottom_x = None
            smoothed_right_bottom_x = None

            smoothed_left_top_x = None
            smoothed_right_top_x = None

            if smoothed_left is not None:

                smoothed_left_bottom_x = smoothed_left[0]
                smoothed_left_top_x = smoothed_left[2]

            if smoothed_right is not None:

                smoothed_right_bottom_x = smoothed_right[0]
                smoothed_right_top_x = smoothed_right[2]

            # --------------------------------------
            # 6.8 Write CSV row
            # --------------------------------------

            csv_writer.writerow([
                frame_count,
                round(timestamp, 3),
                valid,
                left_segments,
                right_segments,
                left_bottom_x,
                right_bottom_x,
                left_top_x,
                right_top_x,
                bottom_width,
                top_width,
                temporal_accepted,
                temporal_reason,
                smoothed_left_bottom_x,
                smoothed_right_bottom_x,
                smoothed_left_top_x,
                smoothed_right_top_x,
            ])

            # --------------------------------------
            # 6.9 Write annotated video frame
            # --------------------------------------

            writer.write(annotated_frame)

            # --------------------------------------
            # 6.10 Print progress
            # --------------------------------------

            if frame_count % 100 == 0:

                print(
                    f"Processed {frame_count} frames | "
                    f"Geometry valid: "
                    f"{geometrically_valid_frames} | "
                    f"Temporal accepted: "
                    f"{temporally_accepted_frames}"
                )

finally:
    cap.release()
    writer.release()


# --------------------------------------------------
# 7. Geometric validation summary
# --------------------------------------------------

total_runtime = time.perf_counter() - start_total

print("\n--- GEOMETRIC VALIDATION ---")

print(f"Total frames: {frame_count}")

print(
    f"Geometrically valid frames: "
    f"{geometrically_valid_frames}"
)

print(
    f"Geometrically invalid frames: "
    f"{geometrically_invalid_frames}"
)

if frame_count > 0:

    geometric_rate = (
        geometrically_valid_frames / frame_count
    )

    print(
        f"Geometric validation rate: "
        f"{geometric_rate:.2%}"
    )


# --------------------------------------------------
# 8. Temporal filter summary
# --------------------------------------------------

print("\n--- TEMPORAL FILTER ---")

print(
    f"Temporally accepted frames: "
    f"{temporally_accepted_frames}"
)

print(
    f"Temporally rejected frames: "
    f"{temporally_rejected_frames}"
)

if frame_count > 0:

    temporal_rate = (
        temporally_accepted_frames / frame_count
    )

    print(
        f"Temporal acceptance rate: "
        f"{temporal_rate:.2%}"
    )

    print(
        f"Geometrically valid but temporally rejected: "
        f"{geometrically_valid_frames - temporally_accepted_frames}"
    )


# --------------------------------------------------
# 9. Processing performance
# --------------------------------------------------

print("\n--- PERFORMANCE ---")

if frame_count > 0:

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
        f"Average lane detection time: "
        f"{avg_detection_time * 1000:.2f} ms/frame"
    )

    print(
        f"Lane detection FPS: "
        f"{detection_fps:.2f}"
    )

    print(
        f"End-to-end throughput: "
        f"{end_to_end_fps:.2f} FPS"
    )


# --------------------------------------------------
# 10. Output files
# --------------------------------------------------

print("\n--- OUTPUT FILES ---")

print(f"Annotated video: {OUTPUT_PATH}")
print(f"Results CSV: {CSV_PATH}")

print("\nProcessing complete!")
