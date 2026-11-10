
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
    get_bottom_center,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
MODEL_PATH = "yolo11n.pt"

TRACKER_PATH = "configs/bytetrack_threshold.yaml"

OUTPUT_PATH = Path("runs/integrated_restored_output.mp4")
CSV_PATH = Path("runs/integrated_restored.csv")

DEVICE = "mps"
CONFIDENCE = 0.25

# COCO: person, bicycle, car, motorcycle, bus, truck
ROAD_USER_CLASSES = [0, 1, 2, 3, 5, 7]

MAX_JUMP = 100
MAX_REFERENCE_AGE = 5
MAX_CANDIDATE_GAP = 2
EMA_ALPHA = 0.3

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Draw accepted lane geometry
# --------------------------------------------------

def draw_lane(frame, left_fit, right_fit):
    """Draw the accepted and smoothed ego lane."""

    annotated = frame.copy()

    if left_fit is None or right_fit is None:
        return annotated

    polygon = build_lane_polygon(left_fit, right_fit)

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [np.rint(polygon).astype(np.int32)],
        (0, 255, 0),
    )

    annotated = cv2.addWeighted(
        overlay,
        0.25,
        frame,
        0.75,
        0,
    )

    cv2.line(
        annotated,
        (left_fit[0], left_fit[1]),
        (left_fit[2], left_fit[3]),
        (0, 255, 0),
        5,
    )

    cv2.line(
        annotated,
        (right_fit[0], right_fit[1]),
        (right_fit[2], right_fit[3]),
        (255, 0, 0),
        5,
    )

    return annotated


# --------------------------------------------------
# 3. Open video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video metadata")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video")


# --------------------------------------------------
# 4. Initialize components
# --------------------------------------------------

model = YOLO(MODEL_PATH)

lane_filter = TemporalLaneFilter(
    max_jump=MAX_JUMP,
    max_reference_age=MAX_REFERENCE_AGE,
    max_candidate_gap=MAX_CANDIDATE_GAP,
)

lane_smoother = LaneSmoother(alpha=EMA_ALPHA)

frame_count = 0
accepted_lane_frames = 0
tracked_observations = 0

status_counts = Counter()
reason_counts = Counter()

start_time = time.perf_counter()

print("\n--- ROADVISION INTEGRATED PIPELINE ---")
print(f"Video: {VIDEO_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Tracker: {TRACKER_PATH}")
print(f"Device: {DEVICE}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps:.1f}")


# --------------------------------------------------
# 5. Process video
# --------------------------------------------------

try:
    with CSV_PATH.open("w", newline="") as csv_file:

        csv_writer = csv.writer(csv_file)

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
            "lane_valid",
            "lane_geometry_valid",
            "lane_temporal_reason",
        ])

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1
            timestamp = (frame_count - 1) / fps

            # --------------------------------------
            # 5.1 Lane detection
            # --------------------------------------

            _, lane_info = detect_lane(frame)

            temporal_accepted, temporal_reason = (
                lane_filter.update(frame_count, lane_info)
            )

            smoothed_left, smoothed_right = (
                lane_smoother.update(
                    lane_info,
                    temporal_accepted,
                    temporal_reason,
                )
            )

            lane_polygon = None

            if temporal_accepted:

                accepted_lane_frames += 1

                lane_polygon = build_lane_polygon(
                    smoothed_left,
                    smoothed_right,
                )

                annotated = draw_lane(
                    frame,
                    smoothed_left,
                    smoothed_right,
                )

            else:
                annotated = frame.copy()

            # --------------------------------------
            # 5.2 YOLO + ByteTrack
            # --------------------------------------

            results = model.track(
                frame,
                persist=True,
                tracker=str(TRACKER_PATH),
                conf=CONFIDENCE,
                classes=ROAD_USER_CLASSES,
                device=DEVICE,
                verbose=False,
            )

            boxes = results[0].boxes

            if boxes is not None and boxes.id is not None:

                coords = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                # ----------------------------------
                # 5.3 Associate tracked objects
                # ----------------------------------

                for bbox, track_id, cls_id, conf in zip(
                    coords,
                    ids,
                    classes,
                    confidences,
                ):

                    x1, y1, x2, y2 = map(float, bbox)

                    bbox_tuple = (x1, y1, x2, y2)

                    status, reason = (
                        classify_vehicle_lane_detailed(
                            bbox_tuple,
                            lane_polygon,
                        )
                    )

                    contact_x, contact_y = (
                        get_bottom_center(bbox_tuple)
                    )

                    class_name = model.names[cls_id]

                    status_counts[status] += 1
                    reason_counts[reason] += 1
                    tracked_observations += 1

                    # ------------------------------
                    # Visualization
                    # ------------------------------

                    colors = {
                        "IN_LANE": (0, 255, 0),
                        "OUT_LANE": (0, 165, 255),
                        "UNKNOWN": (160, 160, 160),
                    }

                    color = colors[status]

                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )

                    label = (
                        f"ID {track_id} {class_name} "
                        f"| {status}"
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), max(int(y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    cv2.circle(
                        annotated,
                        (int(contact_x), int(contact_y)),
                        4,
                        color,
                        -1,
                    )

                    # ------------------------------
                    # CSV logging
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
                        temporal_accepted,
                        lane_info["valid"],
                        temporal_reason,
                    ])

            # --------------------------------------
            # 5.4 Frame summary overlay
            # --------------------------------------

            lane_text = (
                "LANE: AVAILABLE"
                if temporal_accepted
                else "LANE: UNAVAILABLE"
            )

            cv2.putText(
                annotated,
                lane_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0)
                if temporal_accepted
                else (0, 0, 255),
                2,
            )

            writer.write(annotated)

            if frame_count % 100 == 0:
                print(
                    f"Frames: {frame_count} | "
                    f"Lane available: {accepted_lane_frames} | "
                    f"Tracked observations: {tracked_observations}"
                )

finally:
    cap.release()
    writer.release()


# --------------------------------------------------
# 6. Final statistics
# --------------------------------------------------

elapsed = time.perf_counter() - start_time

print("\n--- INTEGRATED PIPELINE RESULTS ---")

print(f"Frames processed: {frame_count}")
print(f"Tracked observations: {tracked_observations}")

if frame_count > 0:
    print(
        f"Lane availability: "
        f"{accepted_lane_frames / frame_count:.2%}"
    )

    print(
        f"End-to-end throughput: "
        f"{frame_count / elapsed:.2f} FPS"
    )

print("\nLane association counts:")

for status, count in status_counts.items():
    print(f"{status}: {count}")

print("\nClassification reasons:")

for reason, count in reason_counts.items():
    print(f"{reason}: {count}")

print(f"\nOutput video: {OUTPUT_PATH}")
print(f"Output CSV: {CSV_PATH}")

print("\nProcessing complete!")
