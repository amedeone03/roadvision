
from pathlib import Path
import math

import numpy as np
import pandas as pd

from src.motion.distance_estimator import (
    CameraModel,
    estimate_ground_distance,
)


INPUT_PATH = Path(
    "runs/motion/robust_motion_results.csv"
)

OUTPUT_PATH = Path(
    "runs/motion/vp_distance_scenarios.csv"
)

# Hypothetical camera parameters, NOT measured.
HEIGHT_M = 1.5
FY_PX = 1000.0
CY_PX = 540.0

HORIZONS = {
    "vp_150": 646.25,
    "vp_mean": 640.23,
    "vp_1023": 634.21,
}

SELECTED_TRACKS = [2, 8, 63, 66, 125, 127]


def camera_from_horizon(horizon_y):
    """
    Construct a hypothetical camera model using
    an image-derived horizon row.

    The principal point, focal length and camera
    height remain uncalibrated.
    """

    pitch_deg = math.degrees(
        math.atan(
            (CY_PX - horizon_y) / FY_PX
        )
    )

    return CameraModel(
        height_m=HEIGHT_M,
        fy_px=FY_PX,
        cy_px=CY_PX,
        pitch_deg=pitch_deg,
    )


df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise RuntimeError("Empty tracking dataset")

required = {
    "frame",
    "track_id",
    "contact_y",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

print("\n--- ROADVISION VANISHING-POINT DISTANCE ---")

for name, horizon in HORIZONS.items():

    camera = camera_from_horizon(horizon)

    print(
        f"{name}: "
        f"horizon={horizon:.2f} px, "
        f"assumed pitch={camera.pitch_deg:.2f} deg"
    )

    values = [
        estimate_ground_distance(y, camera)
        if pd.notna(y)
        else None
        for y in df["contact_y"]
    ]

    df[f"distance_{name}_m"] = pd.Series(
        values, dtype=float
    )


# ------------------------------------------------
# Compare horizon scenarios
# ------------------------------------------------

columns = [
    f"distance_{name}_m"
    for name in HORIZONS
]

df["all_vp_scenarios_valid"] = (
    df[columns].notna().all(axis=1)
)

df["vp_distance_min_m"] = df[
    columns
].min(axis=1)

df["vp_distance_max_m"] = df[
    columns
].max(axis=1)

df["vp_distance_range_m"] = (
    df["vp_distance_max_m"]
    - df["vp_distance_min_m"]
)

reference = df["distance_vp_mean_m"]

df["vp_relative_range"] = (
    df["vp_distance_range_m"] / reference
)

# Purely diagnostic threshold.
df["high_vp_sensitivity"] = (
    df["all_vp_scenarios_valid"]
    & (df["vp_relative_range"] > 0.20)
)


# ------------------------------------------------
# Global summary
# ------------------------------------------------

print(f"\nObservations: {len(df)}")

for name in HORIZONS:

    column = f"distance_{name}_m"

    print(
        f"{name}: "
        f"valid={df[column].notna().sum()}, "
        f"median={df[column].median():.2f} m"
    )

print(
    "All VP scenarios valid:",
    int(df["all_vp_scenarios_valid"].sum())
)

print(
    "High VP sensitivity:",
    int(df["high_vp_sensitivity"].sum())
)


# ------------------------------------------------
# Selected tracks
# ------------------------------------------------

print("\n--- SELECTED TRACKS ---")

for track_id in SELECTED_TRACKS:

    track = df[
        df["track_id"] == track_id
    ]

    if track.empty:
        continue

    print(f"\nTrack {track_id}")
    print(f"Observations: {len(track)}")

    print(
        "Median contact Y: "
        f"{track['contact_y'].median():.2f} px"
    )

    for name in HORIZONS:

        column = f"distance_{name}_m"

        print(
            f"  {name}: "
            f"valid={track[column].notna().sum()}, "
            f"median={track[column].median():.2f} m"
        )

    print(
        "  High VP sensitivity:",
        int(track["high_vp_sensitivity"].sum())
    )


# ------------------------------------------------
# Save
# ------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved: {OUTPUT_PATH}")
print("\nExperiment complete!")
