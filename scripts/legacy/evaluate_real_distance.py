
from pathlib import Path

import numpy as np
import pandas as pd

from src.motion.distance_estimator import (
    CameraModel,
    estimate_ground_distance,
)


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = Path("runs/motion/robust_motion_results.csv")
OUTPUT_PATH = Path("runs/motion/distance_scenarios.csv")

# Hypothetical camera parameters.
# NOT calibrated to the RoadVision video.
CAMERAS = {
    "pitch_2": CameraModel(
        height_m=1.5,
        fy_px=1000,
        cy_px=540,
        pitch_deg=2,
    ),
    "pitch_3": CameraModel(
        height_m=1.5,
        fy_px=1000,
        cy_px=540,
        pitch_deg=3,
    ),
    "pitch_4": CameraModel(
        height_m=1.5,
        fy_px=1000,
        cy_px=540,
        pitch_deg=4,
    ),
}

SELECTED_TRACKS = [2, 8, 63, 66, 125]


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_PATH)

required = {
    "frame",
    "track_id",
    "contact_y",
    "lane_status",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)


# ==========================================
# ESTIMATE DISTANCE FOR EACH SCENARIO
# ==========================================

print("\n--- ROADVISION DISTANCE SCENARIOS ---")
print("Camera parameters are hypothetical.")

for name, camera in CAMERAS.items():

    print(
        f"{name}: horizon_y="
        f"{camera.horizon_y:.2f} px"
    )

    estimates = [
        estimate_ground_distance(y, camera)
        if pd.notna(y) else None
        for y in df["contact_y"]
    ]

    df[f"distance_{name}_m"] = pd.Series(
        estimates, dtype=float
    )


# ==========================================
# CALIBRATION SENSITIVITY
# ==========================================

distance_columns = [
    f"distance_{name}_m"
    for name in CAMERAS
]

# A range across hypothetical camera scenarios,
# not a statistical confidence interval.
df["distance_min_m"] = df[
    distance_columns
].min(axis=1, skipna=True)

df["distance_max_m"] = df[
    distance_columns
].max(axis=1, skipna=True)

df["distance_range_m"] = (
    df["distance_max_m"] - df["distance_min_m"]
)

df["relative_distance_range"] = (
    df["distance_range_m"]
    / df["distance_pitch_3_m"]
)

# Compare only rows where every scenario
# produces a valid forward ground intersection.
df["all_scenarios_valid"] = (
    df[distance_columns].notna().all(axis=1)
)

# Illustrative diagnostic threshold only.
# 20% spread is NOT a calibrated safety limit.
df["high_calibration_sensitivity"] = (
    df["all_scenarios_valid"]
    & (df["relative_distance_range"] > 0.20)
)


# ==========================================
# TEMPORAL DISTANCE CHANGES
# ==========================================

# Differences are computed only when
# observations belong to consecutive frames.
previous_frame = (
    df.groupby("track_id")["frame"].shift(1)
)

consecutive = (
    df["frame"] - previous_frame == 1
)

for name in CAMERAS:

    column = f"distance_{name}_m"

    previous_distance = (
        df.groupby("track_id")[column].shift(1)
    )

    df[f"delta_{name}_m"] = (
        df[column] - previous_distance
    ).where(
        consecutive
        & df[column].notna()
        & previous_distance.notna()
    )


# ==========================================
# GLOBAL SUMMARY
# ==========================================

print(f"\nObservations: {len(df)}")
print(f"Unique tracks: {df['track_id'].nunique()}")

for name in CAMERAS:

    column = f"distance_{name}_m"

    print(
        f"{name}: "
        f"valid={df[column].notna().sum()}, "
        f"median={df[column].median():.2f} m"
    )

print(
    "\nAll scenarios valid:",
    int(df["all_scenarios_valid"].sum()),
)

print(
    "High calibration sensitivity:",
    int(df["high_calibration_sensitivity"].sum()),
)


# ==========================================
# SELECTED TRACK ANALYSIS
# ==========================================

print("\n--- SELECTED TRACKS ---")

for track_id in SELECTED_TRACKS:

    track = df[df["track_id"] == track_id]

    if track.empty:
        continue

    print(f"\nTrack {track_id}")
    print(f"Observations: {len(track)}")
    print(
        "Median contact Y: "
        f"{track['contact_y'].median():.2f} px"
    )

    for name in CAMERAS:

        column = f"distance_{name}_m"
        changes = f"delta_{name}_m"

        print(
            f"  {name}: "
            f"median={track[column].median():.2f} m, "
            f"p95_abs_delta="
            f"{track[changes].abs().quantile(0.95):.2f} m"
        )

    print(
        "  High sensitivity observations:",
        int(track["high_calibration_sensitivity"].sum()),
    )


# ==========================================
# SAVE
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved: {OUTPUT_PATH}")
print("\nDistance scenario analysis complete!")
