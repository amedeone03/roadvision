
from pathlib import Path
from collections import Counter
import csv
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.lanes.lane_detector import detect_lane
from src.lanes.temporal_filter import TemporalLaneFilter
from src.lanes.lane_smoother import LaneSmoother

from src.association.vehicle_lane import (
    build_lane_polygon,
    classify_vehicle_lane_detailed,
    classify_vehicle_lane_conservative,
    get_bottom_center,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

VIDEO_PATH = Path("data/videos/test_drive.mp4")

MODEL_PATH = "yolo11n.pt"
TRACKER_PATH = "configs/bytetrack_threshold.yaml"

OUTPUT_PATH = Path("runs/integrated_final_output.mp4")
CSV_PATH = Path("runs/integrated_final.csv")

DEVICE = "mps"
CONFIDENCE = 0.25

# COCO classes:
# person, bicycle, car, motorcycle, bus, truck
ROAD_USER_CLASSES = [0, 1, 2, 3, 5, 7]

# Temporal lane-filter parameters
MAX_JUMP = 100
MAX_REFERENCE_AGE = 5
MAX_CANDIDATE_GAP = 2

# Lane smoothing
EMA_ALPHA = 0.3

# Conservative association threshold
LATERAL_MARGIN = 0.05

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# 2. VISUALIZATION HELPERS
# ==================================================

def draw_lane(frame, left_fit, right_fit):
    """
    Draw the accepted, smoothed ego-lane polygon.

    Left boundary: green
    Right boundary: blue
    Lane surface: transparent green
    """

    annotated = frame.copy()

    if left_fit is None or right_fit is None:
        return annotated

    polygon = build_lane_polygon(
        left_fit,
        right_fit,
    )

    if polygon is None:
        return annotated

    overlay = annotated.copy()

    cv2.fillPoly(
        overlay,
        [np.rint(polygon).astype(np.int32)],
        (0, 255, 0),
    )

    annotated = cv2.addWeighted(
        overlay,
        0.25,
        annotated,
        0.75,
        0,
    )

    cv2.line(
        annotated,
        (int(left_fit[0]), int(left_fit[1])),
        (int(left_fit[2]), int(left_fit[3])),
        (0, 255, 0),
        5,
    )

    cv2.line(
        annotated,
        (int(right_fit[0]), int(right_fit[1])),
        (int(right_fit[2]), int(right_fit[3])),
        (255, 0, 0),
        5,
    )

    return annotated


def draw_vehicle(
    frame,
    bbox,
    track_id,
    class_name,
    status,
    reason,
):
    """
    Draw a tracked vehicle with its lane status.

    Green: IN_LANE
    Orange: OUT_LANE
    Gray: UNKNOWN
    """

    x1, y1, x2, y2 = map(float, bbox)

    colors = {
        "IN_LANE": (0, 255, 0),
        "OUT_LANE": (0, 165, 255),
        "UNKNOWN": (160, 160, 160),
    }

    color = colors.get(
        status,
        (255, 255, 255),
    )

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2,
    )

    label = (
        f"ID {track_id} {class_name} | {status}"
    )

    cv2.putText(
        frame,
        label,
        (int(x1), max(int(y1) - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    # Indicate the location of the contact point.
    contact_x, contact_y = get_bottom_center(bbox)

    cv2.circle(
        frame,
        (int(contact_x), int(contact_y)),
        4,
        color,
        -1,
    )

    # Explain ambiguous classifications.
    if status == "UNKNOWN":

        reason_text = reason.replace("_", " ")

        cv2.putText(
            frame,
            reason_text,
            (int(x1), min(int(y2) + 20, frame.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


# ==================================================
# 3. INITIALIZE VIDEO
# ==================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if not Path(TRACKER_PATH).exists():
    raise FileNotFoundError(
        f"Tracker configuration not found: {TRACKER_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError(
        "Invalid input video metadata"
    )

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create video: {OUTPUT_PATH}"
    )


# ==================================================
# 4. INITIALIZE PERCEPTION COMPONENTS
# ==================================================

model = YOLO(MODEL_PATH)

lane_filter = TemporalLaneFilter(
    max_jump=MAX_JUMP,
    max_reference_age=MAX_REFERENCE_AGE,
    max_candidate_gap=MAX_CANDIDATE_GAP,
)

lane_smoother = LaneSmoother(
    alpha=EMA_ALPHA,
)

frame_count = 0
accepted_lane_frames = 0
tracked_observations = 0

status_counts = Counter()
reason_counts = Counter()

baseline_status_counts = Counter()
changes_by_track = Counter()

start_time = time.perf_counter()

print("\n" + "=" * 55)
print("ROADVISION — FINAL INTEGRATED PIPELINE")
print("=" * 55)

print(f"Input: {VIDEO_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Tracker: {TRACKER_PATH}")
print(f"Device: {DEVICE}")

print(f"Resolution: {width} x {height}")
print(f"Input FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")

print(f"Lane EMA alpha: {EMA_ALPHA}")
print(f"Lateral uncertainty margin: {LATERAL_MARGIN:.0%}")

print(f"\nOutput video: {OUTPUT_PATH}")
print(f"Output CSV: {CSV_PATH}")


# ==================================================
# 5. PROCESS VIDEO
# ==================================================

try:

    with CSV_PATH.open(
        "w",
        newline="",
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        # ------------------------------------------
        # CSV HEADER
        # ------------------------------------------

        csv_writer.writerow([
            "frame",
            "timestamp",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "contact_x",
            "contact_y",
            "lane_status",
            "lane_reason",
            "baseline_status",
            "baseline_reason",
            "normalized_lateral_margin",
            "lane_valid",
            "lane_geometry_valid",
            "lane_temporal_reason",
        ])

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            timestamp = (
                (frame_count - 1) / fps
            )

            # ======================================
            # 5.1 GEOMETRIC LANE DETECTION
            # ======================================

            _, lane_info = detect_lane(frame)

            geometry_valid = bool(
                lane_info["valid"]
            )

            # ======================================
            # 5.2 TEMPORAL LANE FILTER
            # ======================================

            temporal_accepted, temporal_reason = (
                lane_filter.update(
                    frame_count,
                    lane_info,
                )
            )

            if temporal_accepted:
                accepted_lane_frames += 1

            # ======================================
            # 5.3 LANE SMOOTHING
            # ======================================

            smoothed_left, smoothed_right = (
                lane_smoother.update(
                    lane_info,
                    temporal_accepted,
                    temporal_reason,
                )
            )

            # Only use a lane polygon when
            # this frame's geometry was accepted.
            lane_polygon = None

            if (
                temporal_accepted
                and smoothed_left is not None
                and smoothed_right is not None
            ):

                lane_polygon = build_lane_polygon(
                    smoothed_left,
                    smoothed_right,
                )

            # ======================================
            # 5.4 DRAW EGO LANE
            # ======================================

            annotated = draw_lane(
                frame,
                smoothed_left if temporal_accepted else None,
                smoothed_right if temporal_accepted else None,
            )

            # ======================================
            # 5.5 YOLO + BYTETRACK
            # ======================================

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_PATH,
                conf=CONFIDENCE,
                classes=ROAD_USER_CLASSES,
                device=DEVICE,
                verbose=False,
            )

            boxes = results[0].boxes

            if (
                boxes is not None
                and boxes.id is not None
            ):

                coords = (
                    boxes.xyxy.cpu().numpy()
                )

                ids = (
                    boxes.id.int().cpu().tolist()
                )

                classes = (
                    boxes.cls.int().cpu().tolist()
                )

                confidences = (
                    boxes.conf.cpu().tolist()
                )

                # ==================================
                # 5.6 VEHICLE-TO-LANE ASSOCIATION
                # ==================================

                for bbox, track_id, cls_id, conf in zip(
                    coords,
                    ids,
                    classes,
                    confidences,
                ):

                    x1, y1, x2, y2 = map(
                        float,
                        bbox,
                    )

                    bbox_tuple = (
                        x1,
                        y1,
                        x2,
                        y2,
                    )

                    class_name = model.names[
                        cls_id
                    ]

                    contact_x, contact_y = (
                        get_bottom_center(
                            bbox_tuple
                        )
                    )

                    # ------------------------------
                    # Original baseline
                    # ------------------------------

                    baseline_status, baseline_reason = (
                        classify_vehicle_lane_detailed(
                            bbox_tuple,
                            lane_polygon,
                        )
                    )

                    baseline_status_counts[
                        baseline_status
                    ] += 1

                    # ------------------------------
                    # Conservative classifier
                    # ------------------------------

                    status, reason, normalized_margin = (
                        classify_vehicle_lane_conservative(
                            bbox_tuple,
                            smoothed_left
                            if temporal_accepted
                            else None,
                            smoothed_right
                            if temporal_accepted
                            else None,
                            margin=LATERAL_MARGIN,
                        )
                    )

                    status_counts[status] += 1
                    reason_counts[reason] += 1

                    tracked_observations += 1

                    # Count changed classifications.
                    if status != baseline_status:

                        changes_by_track[
                            track_id
                        ] += 1

                    # ------------------------------
                    # Draw tracked vehicle
                    # ------------------------------

                    draw_vehicle(
                        annotated,
                        bbox_tuple,
                        track_id,
                        class_name,
                        status,
                        reason,
                    )

                    # ------------------------------
                    # Write observation to CSV
                    # ------------------------------

                    csv_writer.writerow([
                        frame_count,
                        round(timestamp, 3),
                        track_id,
                        cls_id,
                        class_name,
                        round(float(conf), 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        round(contact_x, 2),
                        round(contact_y, 2),
                        status,
                        reason,
                        baseline_status,
                        baseline_reason,
                        (
                            round(normalized_margin, 5)
                            if normalized_margin is not None
                            else ""
                        ),
                        temporal_accepted,
                        geometry_valid,
                        temporal_reason,
                    ])

            # ======================================
            # 5.7 FRAME SUMMARY OVERLAY
            # ======================================

            lane_text = (
                "LANE: AVAILABLE"
                if temporal_accepted
                else "LANE: UNAVAILABLE"
            )

            lane_color = (
                (0, 255, 0)
                if temporal_accepted
                else (0, 0, 255)
            )

            cv2.putText(
                annotated,
                lane_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                lane_color,
                2,
            )

            cv2.putText(
                annotated,
                f"FRAME: {frame_count}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"LANE MARGIN: {LATERAL_MARGIN:.0%}",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            writer.write(annotated)

            # ======================================
            # 5.8 PROGRESS
            # ======================================

            if frame_count % 100 == 0:

                print(
                    f"Frame {frame_count}/{total_frames} | "
                    f"Lane accepted: {accepted_lane_frames} | "
                    f"Tracks: {tracked_observations}"
                )

finally:
    cap.release()
    writer.release()


# ==================================================
# 6. FINAL PERFORMANCE STATISTICS
# ==================================================

elapsed = (
    time.perf_counter() - start_time
)

print("\n" + "=" * 55)
print("ROADVISION — FINAL RESULTS")
print("=" * 55)

print(f"Frames processed: {frame_count}")

print(
    f"Tracked observations: "
    f"{tracked_observations}"
)

print(
    f"Accepted lane frames: "
    f"{accepted_lane_frames}"
)

if frame_count > 0:

    lane_availability = (
        accepted_lane_frames / frame_count
    )

    throughput = (
        frame_count / elapsed
    )

    print(
        f"Lane availability: "
        f"{lane_availability:.2%}"
    )

    print(
        f"End-to-end throughput: "
        f"{throughput:.2f} FPS"
    )


print("\n--- BASELINE CLASSIFICATIONS ---")

for status in [
    "IN_LANE",
    "OUT_LANE",
    "UNKNOWN",
]:
    print(
        f"{status}: "
        f"{baseline_status_counts[status]}"
    )


print("\n--- CONSERVATIVE CLASSIFICATIONS ---")

for status in [
    "IN_LANE",
    "OUT_LANE",
    "UNKNOWN",
]:
    print(
        f"{status}: "
        f"{status_counts[status]}"
    )


print("\n--- CLASSIFICATION REASONS ---")

for reason, count in reason_counts.most_common():
    print(f"{reason}: {count}")


print("\n--- CHANGED CLASSIFICATIONS ---")

total_changes = sum(
    changes_by_track.values()
)

print(f"Total changed observations: {total_changes}")

for track_id, count in changes_by_track.most_common():
    print(
        f"Track {track_id}: {count} changes"
    )


print("\n--- OUTPUT FILES ---")
print(f"Video: {OUTPUT_PATH}")
print(f"CSV: {CSV_PATH}")

print("\nProcessing complete!")
