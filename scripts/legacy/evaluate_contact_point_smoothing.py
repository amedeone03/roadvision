
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from src.lanes.lane_detector import detect_lane
from src.lanes.temporal_filter import TemporalLaneFilter
from src.lanes.lane_smoother import LaneSmoother
from src.association.vehicle_lane import build_lane_polygon


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
CSV_PATH = Path("runs/integrated_restored.csv")

OUTPUT_DIR = Path("runs/association_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "contact_smoothing_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "contact_smoothing_summary.csv"

ALPHAS = {
    "RAW": None,
    "EMA_03": 0.3,
    "EMA_06": 0.6,
}

# Visually inspected adjacent-lane truck sequences.
TRUCK_WINDOWS = {
    8: (295, 330),
    125: (995, 1025),
}

REFERENCE_CAR_ID = 2


# --------------------------------------------------
# 2. Contact-point EMA
# --------------------------------------------------

class VehiclePointSmoother:

    def __init__(self, alpha):
        self.alpha = alpha
        self.states = {}

    def update(self, track_id, frame_number, point):
        """
        Smooth the contact point separately for each track.

        Reset if the track was not observed
        in the immediately previous frame.
        """

        point = np.asarray(point, dtype=float)

        if track_id not in self.states:
            smoothed = point.copy()

        else:
            previous_frame, previous_point = (
                self.states[track_id]
            )

            if frame_number != previous_frame + 1:
                smoothed = point.copy()
            else:
                smoothed = (
                    self.alpha * point
                    + (1 - self.alpha) * previous_point
                )

        self.states[track_id] = (
            frame_number,
            smoothed.copy(),
        )

        return smoothed


# --------------------------------------------------
# 3. Classify a contact point
# --------------------------------------------------

def classify_point(point, polygon):
    """
    Reproduce the baseline lane classification,
    but accept a contact point directly.
    """

    if polygon is None:
        return "UNKNOWN", "lane_unavailable", np.nan

    x, y = map(float, point)

    y_min = float(polygon[:, 1].min())
    y_max = float(polygon[:, 1].max())

    if y < y_min:
        return "UNKNOWN", "above_lane_coverage", np.nan

    if y > y_max:
        return "UNKNOWN", "below_lane_coverage", np.nan

    distance = cv2.pointPolygonTest(
        polygon,
        (x, y),
        True,
    )

    if distance >= 0:
        return "IN_LANE", "inside_polygon", distance

    return "OUT_LANE", "outside_polygon", distance


# --------------------------------------------------
# 4. Count consecutive-frame direct flips
# --------------------------------------------------

def count_direct_flips(df, status_column):
    """
    Count IN_LANE <-> OUT_LANE transitions
    occurring on consecutive frames.
    """

    ordered = df.sort_values(
        ["track_id", "frame"]
    ).copy()

    previous_frame = (
        ordered.groupby("track_id")["frame"].shift()
    )

    previous_status = (
        ordered.groupby("track_id")[status_column].shift()
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
# 5. Load tracked detections
# --------------------------------------------------

if not VIDEO_PATH.exists():
    raise FileNotFoundError(VIDEO_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

tracks = pd.read_csv(CSV_PATH)

required = {
    "frame",
    "track_id",
    "contact_x",
    "contact_y",
    "lane_status",
    "lane_reason",
}

missing = required - set(tracks.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if tracks.duplicated(["frame", "track_id"]).any():
    raise ValueError("Duplicate track observations found")

tracks = tracks.sort_values(["frame", "track_id"])

observations_by_frame = {
    int(frame): group
    for frame, group in tracks.groupby("frame")
}

print("\n--- CONTACT-POINT SMOOTHING EXPERIMENT ---")
print(f"Tracked observations: {len(tracks)}")


# --------------------------------------------------
# 6. Initialize lane pipeline and smoothers
# --------------------------------------------------

lane_filter = TemporalLaneFilter(
    max_jump=100,
    max_reference_age=5,
    max_candidate_gap=2,
)

lane_smoother = LaneSmoother(alpha=0.3)

point_smoothers = {
    name: VehiclePointSmoother(alpha)
    for name, alpha in ALPHAS.items()
    if alpha is not None
}

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video")

results = []

frame_number = 0
accepted_frames = 0


# --------------------------------------------------
# 7. Replay the complete lane pipeline
# --------------------------------------------------

try:
    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Lane pipeline must process EVERY frame,
        # including frames without tracked objects.
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

        # No tracked observations in this frame.
        if frame_number not in observations_by_frame:
            continue

        frame_objects = observations_by_frame[frame_number]

        for obj in frame_objects.itertuples(index=False):

            track_id = int(obj.track_id)

            raw_point = np.array(
                [obj.contact_x, obj.contact_y],
                dtype=float,
            )

            row = {
                "frame": frame_number,
                "track_id": track_id,
                "baseline_status": obj.lane_status,
                "baseline_reason": obj.lane_reason,
                "raw_x": raw_point[0],
                "raw_y": raw_point[1],
                "lane_available": accepted,
            }

            # Evaluate raw and smoothed coordinates
            # against the same accepted lane polygon.
            for method, alpha in ALPHAS.items():

                if alpha is None:
                    point = raw_point.copy()
                else:
                    point = point_smoothers[
                        method
                    ].update(
                        track_id,
                        frame_number,
                        raw_point,
                    )

                status, reason, distance = classify_point(
                    point,
                    polygon,
                )

                row[f"{method}_x"] = point[0]
                row[f"{method}_y"] = point[1]
                row[f"{method}_status"] = status
                row[f"{method}_reason"] = reason
                row[f"{method}_distance"] = distance

            results.append(row)

        if frame_number % 200 == 0:
            print(f"Processed {frame_number} frames")

finally:
    cap.release()


# --------------------------------------------------
# 8. Export results
# --------------------------------------------------

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No track observations processed")

results_df.to_csv(RESULTS_PATH, index=False)

print(f"\nTotal video frames: {frame_number}")
print(f"Accepted lane frames: {accepted_frames}")

# Validate the reconstructed raw baseline.
baseline_mismatch = (
    results_df["RAW_status"]
    != results_df["baseline_status"]
)

reason_mismatch = (
    results_df["RAW_reason"]
    != results_df["baseline_reason"]
)

print("\n--- BASELINE REPRODUCTION ---")
print(f"Status mismatches: {baseline_mismatch.sum()}")
print(f"Reason mismatches: {reason_mismatch.sum()}")

if baseline_mismatch.any() or reason_mismatch.any():
    print(
        "WARNING: The reconstructed lane pipeline "
        "does not exactly match the saved baseline. "
        "Check detector configuration before "
        "interpreting the experiment."
    )


# --------------------------------------------------
# 9. Compare methods globally
# --------------------------------------------------

summary_rows = []

print("\n--- GLOBAL METHOD COMPARISON ---")

for method in ALPHAS:

    column = f"{method}_status"

    counts = results_df[column].value_counts()

    flips = count_direct_flips(
        results_df,
        column,
    )

    row = {
        "method": method,
        "in_lane": int(counts.get("IN_LANE", 0)),
        "out_lane": int(counts.get("OUT_LANE", 0)),
        "unknown": int(counts.get("UNKNOWN", 0)),
        "direct_flips": flips,
    }

    summary_rows.append(row)

    print(
        f"{method}: "
        f"IN={row['in_lane']}, "
        f"OUT={row['out_lane']}, "
        f"UNKNOWN={row['unknown']}, "
        f"flips={flips}"
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_PATH, index=False)


# --------------------------------------------------
# 10. Evaluate the two problematic truck windows
# --------------------------------------------------

print("\n--- ADJACENT-LANE TRUCK WINDOWS ---")

for track_id, (start, end) in TRUCK_WINDOWS.items():

    window = results_df[
        (results_df["track_id"] == track_id)
        & results_df["frame"].between(start, end)
    ]

    print(
        f"\nTrack {track_id}, "
        f"frames {start}-{end}: "
        f"{len(window)} observations"
    )

    for method in ALPHAS:

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


# --------------------------------------------------
# 11. Compare performance on the car ahead
# --------------------------------------------------

print("\n--- REFERENCE CAR: TRACK 2 ---")

car = results_df[
    results_df["track_id"] == REFERENCE_CAR_ID
]

for method in ALPHAS:

    column = f"{method}_status"

    in_count = int(
        (car[column] == "IN_LANE").sum()
    )

    flips = count_direct_flips(car, column)

    print(
        f"{method}: "
        f"IN_LANE={in_count}, "
        f"direct_flips={flips}"
    )


# --------------------------------------------------
# 12. Finish
# --------------------------------------------------

print("\n--- OUTPUT FILES ---")

print(RESULTS_PATH)
print(SUMMARY_PATH)

print("\nExperiment complete!")
