
from pathlib import Path
import csv
import time

import cv2

from src.lanes.lane_detector import detect_lane
from src.lanes.temporal_filter import TemporalLaneFilter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")

OUTPUT_PATH = Path("runs/lane_temporal_output.mp4")
CSV_PATH = Path("runs/lane_temporal_results.csv")

# Experimental temporal filtering parameters
MAX_JUMP = 100
MAX_REFERENCE_AGE = 5

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


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

print("\n--- ROADVISION TEMPORAL LANE DETECTION ---")
print(f"Input video: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"Input FPS: {fps:.1f}")
print(f"Maximum endpoint jump: {MAX_JUMP} px")
print(f"Maximum reference age: {MAX_REFERENCE_AGE} frames")


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
# 4. Initialize temporal filter and statistics
# --------------------------------------------------

lane_filter = TemporalLaneFilter(
    max_jump=MAX_JUMP,
    max_reference_age=MAX_REFERENCE_AGE,
)

frame_count = 0

valid_lane_frames = 0
invalid_lane_frames = 0

temporally_accepted_frames = 0
temporally_rejected_frames = 0

total_lane_processing_time = 0.0

start_total = time.perf_counter()


# --------------------------------------------------
# 5. Process video and log results
# --------------------------------------------------

try:
    with CSV_PATH.open("w", newline="") as csv_file:

        csv_writer = csv.writer(csv_file)

        # One row per video frame
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
        ])

        while True:

            # --------------------------------------
            # Read frame
            # --------------------------------------

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # CSV frame numbers are 1-based;
            # timestamps start at zero.
            timestamp = (frame_count - 1) / fps

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
            # Geometric validation statistics
            # --------------------------------------

            valid = lane_info["valid"]

            left_fit = lane_info["left_fit"]
            right_fit = lane_info["right_fit"]

            left_segments = lane_info["left_segments"]
            right_segments = lane_info["right_segments"]

            if valid:
                valid_lane_frames += 1
            else:
                invalid_lane_frames += 1

            # --------------------------------------
            # Temporal consistency filtering
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

                # The lane detector may have drawn a
                # geometrically valid polygon before
                # temporal filtering rejected it.
                #
                # Return to the original frame so
                # rejected polygons are not displayed.
                annotated_frame = frame.copy()

            # --------------------------------------
            # Extract raw lane coordinates
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

            # Preserve raw fitted widths even when
            # geometry or temporal filtering fails.
            if left_fit is not None and right_fit is not None:

                bottom_width = (
                    right_bottom_x - left_bottom_x
                )

                top_width = (
                    right_top_x - left_top_x
                )

            # --------------------------------------
            # Save CSV row
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
            ])

            # --------------------------------------
            # Write output frame
            # --------------------------------------

            writer.write(annotated_frame)

            # --------------------------------------
            # Progress reporting
            # --------------------------------------

            if frame_count % 100 == 0:

                print(
                    f"Processed {frame_count} frames | "
                    f"Geometry valid: {valid_lane_frames} | "
                    f"Temporal accepted: "
                    f"{temporally_accepted_frames}"
                )

finally:

    # Release resources even if an error occurs.
    cap.release()
    writer.release()


# --------------------------------------------------
# 6. Performance statistics
# --------------------------------------------------

total_runtime = time.perf_counter() - start_total

print("\n--- GEOMETRIC VALIDATION RESULTS ---")

print(f"Frames processed: {frame_count}")
print(f"Geometrically valid frames: {valid_lane_frames}")
print(f"Geometrically invalid frames: {invalid_lane_frames}")

if frame_count > 0:

    geometric_rate = valid_lane_frames / frame_count

    print(
        f"Geometric validation rate: "
        f"{geometric_rate:.2%}"
    )


# --------------------------------------------------
# 7. Temporal filtering statistics
# --------------------------------------------------

print("\n--- TEMPORAL FILTER RESULTS ---")

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

    temporal_only_rejections = (
        valid_lane_frames - temporally_accepted_frames
    )

    print(
        f"Temporal acceptance rate: "
        f"{temporal_rate:.2%}"
    )

    print(
        f"Geometrically valid frames rejected "
        f"by temporal filter: "
        f"{temporal_only_rejections}"
    )


# --------------------------------------------------
# 8. Processing performance
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


# --------------------------------------------------
# 9. Output locations
# --------------------------------------------------

print("\n--- OUTPUT FILES ---")

print(f"Annotated video: {OUTPUT_PATH}")
print(f"Results CSV: {CSV_PATH}")

print("\nProcessing complete!")
