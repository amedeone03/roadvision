
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.lanes.lane_detector import detect_lane
from src.lanes.temporal_filter import TemporalLaneFilter
from src.lanes.lane_smoother import LaneSmoother
from src.association.vehicle_lane import (
    build_lane_polygon,
    classify_vehicle_lane_detailed,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
CSV_PATH = Path("runs/integrated_restored.csv")

OUTPUT_DIR = Path("runs/association_debug")

RESULTS_PATH = OUTPUT_DIR / "lateral_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "lateral_summary.csv"

MARGINS = {
    "LATERAL_0": 0.00,
    "LATERAL_5": 0.05,
    "LATERAL_10": 0.10,
}

TRUCK_WINDOWS = {
    8: (295, 330),
    125: (995, 1025),
}

REFERENCE_CAR_ID = 2


# --------------------------------------------------
# 2. Interpolate a lane boundary
# --------------------------------------------------

def lane_x_at_y(fit, y):
    """
    Calculate lane-boundary X at image height Y.

    fit:
        (x_bottom, y_bottom, x_top, y_top)

    Uses linear interpolation, not extrapolation.
    """

    x_bottom, y_bottom, x_top, y_top = fit

    if y_bottom == y_top:
        return np.nan

    if not min(y_bottom, y_top) <= y <= max(
        y_bottom, y_top
    ):
        return np.nan

    fraction = (
        (y - y_top) / (y_bottom - y_top)
    )

    return (
        x_top
        + fraction * (x_bottom - x_top)
    )


# --------------------------------------------------
# 3. Calculate lateral measurements
# --------------------------------------------------

def calculate_lateral_metrics(point, left_fit, right_fit):
    """
    Return:
        left boundary X
        right boundary X
        horizontal lane width
        signed distance to left boundary
        signed distance to right boundary
        normalized minimum inside margin
    """

    if left_fit is None or right_fit is None:
        return None

    x, y = map(float, point)

    x_left = lane_x_at_y(left_fit, y)
    x_right = lane_x_at_y(right_fit, y)

    if not np.isfinite(x_left) or not np.isfinite(x_right):
        return None

    lane_width = x_right - x_left

    if lane_width <= 0:
        return None

    d_left = x - x_left
    d_right = x_right - x

    # Positive inside the lane, negative outside.
    normalized_margin = (
        min(d_left, d_right) / lane_width
    )

    return {
        "lane_left_x": x_left,
        "lane_right_x": x_right,
        "lane_width": lane_width,
        "d_left": d_left,
        "d_right": d_right,
        "normalized_margin": normalized_margin,
    }


# --------------------------------------------------
# 4. Experimental lateral classification
# --------------------------------------------------

def classify_lateral(metrics, lane_available, margin):
    """
    Use horizontal lane-boundary distances.

    A point clearly outside remains OUT_LANE.

    A point inside but too close to a boundary
    becomes UNKNOWN.

    No polygon extrapolation is performed.
    """

    if not lane_available:
        return "UNKNOWN", "lane_unavailable"

    if metrics is None:
        return "UNKNOWN", "outside_vertical_coverage"

    d_left = metrics["d_left"]
    d_right = metrics["d_right"]

    if d_left < 0 or d_right < 0:
        return "OUT_LANE", "outside_lateral_boundaries"

    normalized_margin = metrics["normalized_margin"]

    if normalized_margin < margin:
        return "UNKNOWN", "near_lane_boundary"

    return "IN_LANE", "inside_lateral_boundaries"


# --------------------------------------------------
# 5. Count direct lane flips
# --------------------------------------------------

def count_direct_flips(data, status_column):

    ordered = data.sort_values(
        ["track_id", "frame"]
    ).copy()

    previous_frame = (
        ordered.groupby("track_id")["frame"].shift(1)
    )

    previous_status = (
        ordered.groupby("track_id")[status_column]
        .shift(1)
    )

    consecutive = (
        ordered["frame"] - previous_frame == 1
    )

    definite = {"IN_LANE", "OUT_LANE"}

    flips = (
        consecutive
        & ordered[status_column].isin(definite)
        & previous_status.isin(definite)
        & ordered[status_column].ne(previous_status)
    )

    return int(flips.sum())


# --------------------------------------------------
# 6. Load observations
# --------------------------------------------------

if not VIDEO_PATH.exists():
    raise FileNotFoundError(VIDEO_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tracks = pd.read_csv(CSV_PATH)

required = {
    "frame",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "lane_status",
    "lane_reason",
}

missing = required - set(tracks.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

if tracks.duplicated(["frame", "track_id"]).any():
    raise ValueError(
        "Duplicate track observations found"
    )

tracks = tracks.sort_values(
    ["frame", "track_id"]
)

observations_by_frame = {
    int(frame): group
    for frame, group in tracks.groupby("frame")
}

print("\n--- LATERAL ASSOCIATION EXPERIMENT ---")
print(f"Observations: {len(tracks)}")


# --------------------------------------------------
# 7. Initialize lane pipeline
# --------------------------------------------------

lane_filter = TemporalLaneFilter(
    max_jump=100,
    max_reference_age=5,
    max_candidate_gap=2,
)

lane_smoother = LaneSmoother(alpha=0.3)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

results = []
frame_number = 0
accepted_frames = 0


# --------------------------------------------------
# 8. Replay full video
# --------------------------------------------------

try:
    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        _, lane_info = detect_lane(frame)

        accepted, temporal_reason = lane_filter.update(
            frame_number,
            lane_info,
        )

        left_fit, right_fit = lane_smoother.update(
            lane_info,
            accepted,
            temporal_reason,
        )

        polygon = None

        if accepted:
            accepted_frames += 1

            polygon = build_lane_polygon(
                left_fit,
                right_fit,
            )

        frame_objects = observations_by_frame.get(
            frame_number
        )

        if frame_objects is None:
            continue

        for obj in frame_objects.itertuples(index=False):

            bbox = (
                float(obj.x1),
                float(obj.y1),
                float(obj.x2),
                float(obj.y2),
            )

            x1, y1, x2, y2 = bbox

            point = (
                (x1 + x2) / 2,
                y2,
            )

            # Reconstruct the original classification.
            baseline_status, baseline_reason = (
                classify_vehicle_lane_detailed(
                    bbox,
                    polygon,
                )
            )

            metrics = calculate_lateral_metrics(
                point,
                left_fit if accepted else None,
                right_fit if accepted else None,
            )

            row = {
                "frame": frame_number,
                "track_id": int(obj.track_id),
                "class_name": obj.class_name,
                "contact_x": point[0],
                "contact_y": point[1],
                "lane_available": accepted,
                "saved_status": obj.lane_status,
                "saved_reason": obj.lane_reason,
                "RAW_status": baseline_status,
                "RAW_reason": baseline_reason,
            }

            if metrics is not None:
                row.update(metrics)
            else:
                for key in [
                    "lane_left_x",
                    "lane_right_x",
                    "lane_width",
                    "d_left",
                    "d_right",
                    "normalized_margin",
                ]:
                    row[key] = np.nan

            # Evaluate each candidate margin.
            for name, margin in MARGINS.items():

                status, reason = classify_lateral(
                    metrics,
                    accepted,
                    margin,
                )

                row[f"{name}_status"] = status
                row[f"{name}_reason"] = reason

            results.append(row)

        if frame_number % 200 == 0:
            print(f"Processed {frame_number} frames")

finally:
    cap.release()


# --------------------------------------------------
# 9. Validate baseline reproduction
# --------------------------------------------------

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No observations processed")

status_mismatch = (
    df["RAW_status"] != df["saved_status"]
).sum()

reason_mismatch = (
    df["RAW_reason"] != df["saved_reason"]
).sum()

print("\n--- BASELINE REPRODUCTION ---")

print(f"Video frames: {frame_number}")
print(f"Accepted lane frames: {accepted_frames}")
print(f"Status mismatches: {status_mismatch}")
print(f"Reason mismatches: {reason_mismatch}")

df.to_csv(RESULTS_PATH, index=False)

if status_mismatch or reason_mismatch:
    raise RuntimeError(
        "Baseline mismatch: check lane configuration "
        "before interpreting experimental results."
    )


# --------------------------------------------------
# 10. Global comparison
# --------------------------------------------------

print("\n--- GLOBAL COMPARISON ---")

methods = ["RAW"] + list(MARGINS)
summary_rows = []

for method in methods:

    status_column = f"{method}_status"

    counts = df[status_column].value_counts()

    flips = count_direct_flips(
        df,
        status_column,
    )

    summary_rows.append({
        "method": method,
        "in_lane": int(counts.get("IN_LANE", 0)),
        "out_lane": int(counts.get("OUT_LANE", 0)),
        "unknown": int(counts.get("UNKNOWN", 0)),
        "direct_flips": flips,
    })

summary_df = pd.DataFrame(summary_rows)

print(summary_df.to_string(index=False))

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# --------------------------------------------------
# 11. Evaluate problematic truck windows
# --------------------------------------------------

print("\n--- ADJACENT-LANE TRUCK WINDOWS ---")

for track_id, (start, end) in TRUCK_WINDOWS.items():

    window = df[
        (df["track_id"] == track_id)
        & df["frame"].between(start, end)
    ]

    print(
        f"\nTrack {track_id}, frames {start}-{end}"
    )

    for method in methods:

        column = f"{method}_status"

        in_count = int(
            (window[column] == "IN_LANE").sum()
        )

        out_count = int(
            (window[column] == "OUT_LANE").sum()
        )

        unknown_count = int(
            (window[column] == "UNKNOWN").sum()
        )

        flips = count_direct_flips(
            window,
            column,
        )

        print(
            f"  {method}: "
            f"IN={in_count}, "
            f"OUT={out_count}, "
            f"UNKNOWN={unknown_count}, "
            f"flips={flips}"
        )

    # Inspect original false-positive contact points.
    false_inside = window[
        window["RAW_status"] == "IN_LANE"
    ]

    if not false_inside.empty:

        print("\nBaseline false-IN measurements:")

        print(
            false_inside[
                [
                    "frame",
                    "d_left",
                    "d_right",
                    "lane_width",
                    "normalized_margin",
                ]
            ].round(3).to_string(index=False)
        )


# --------------------------------------------------
# 12. Evaluate reference car
# --------------------------------------------------

print("\n--- REFERENCE CAR: TRACK 2 ---")

car = df[
    df["track_id"] == REFERENCE_CAR_ID
]

for method in methods:

    column = f"{method}_status"

    in_count = int(
        (car[column] == "IN_LANE").sum()
    )

    print(
        f"{method}: IN_LANE={in_count}"
    )

# Compare only observations classified IN_LANE
# by the original baseline.
baseline_car_inside = car[
    car["RAW_status"] == "IN_LANE"
]

print("\nRetention of original car IN_LANE predictions:")

for method in methods:

    retained = int(
        (
            baseline_car_inside[
                f"{method}_status"
            ] == "IN_LANE"
        ).sum()
    )

    total = len(baseline_car_inside)

    percentage = (
        100 * retained / total if total else 0
    )

    print(
        f"{method}: {retained}/{total} "
        f"({percentage:.1f}%)"
    )


# --------------------------------------------------
# 13. Final output
# --------------------------------------------------

print("\n--- OUTPUT FILES ---")
print(RESULTS_PATH)
print(SUMMARY_PATH)

print("\nExperiment complete!")