
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.motion.image_motion import (
    bounding_box_features,
    calculate_image_motion,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

INPUT_PATH = Path("runs/integrated_final.csv")
VIDEO_PATH = Path("data/videos/test_drive.mp4")

OUTPUT_DIR = Path("runs/motion")
OUTPUT_PATH = OUTPUT_DIR / "image_motion_results.csv"

# Tracks previously inspected during Phase 4.
SELECTED_TRACKS = [2, 8, 63, 66, 125]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. VIDEO METADATA
# ==================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(VIDEO_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

try:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
finally:
    cap.release()

if not np.isfinite(fps) or fps <= 0:
    raise RuntimeError("Invalid video frame rate")

if width <= 0 or height <= 0:
    raise RuntimeError("Invalid video dimensions")

frame_dt = 1.0 / fps


# ==================================================
# 3. LOAD TRACKING DATA
# ==================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

df = pd.read_csv(INPUT_PATH)

required = {
    "frame",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "lane_status",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

if df.empty:
    raise RuntimeError("Tracking dataset is empty")

if df.duplicated(["track_id", "frame"]).any():
    raise ValueError(
        "Duplicate track/frame observations found"
    )

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)

# Use original video timing rather than rounded
# CSV timestamps for frame-to-frame derivatives.
df["time_seconds"] = (df["frame"] - 1) / fps

print("\n--- ROADVISION IMAGE-SPACE MOTION ---")
print(f"Input: {INPUT_PATH}")
print(f"Observations: {len(df)}")
print(f"Unique tracks: {df['track_id'].nunique()}")
print(f"Video FPS: {fps:.3f}")
print(f"Frame interval: {frame_dt:.5f} s")


# ==================================================
# 4. EXTRACT BOUNDING-BOX FEATURES
# ==================================================

features = []

for row in df.itertuples(index=False):

    bbox = (
        row.x1,
        row.y1,
        row.x2,
        row.y2,
    )

    features.append(
        bounding_box_features(bbox)
    )

features_df = pd.DataFrame(features)

# The integrated CSV already contains contact_x
# and contact_y. Keep those original columns.
features_df = features_df.drop(
    columns=["contact_x", "contact_y"]
)

df = pd.concat(
    [df, features_df],
    axis=1,
)

# Ensure no duplicate columns remain.
assert df.columns.is_unique, (
    "Duplicate columns detected in motion dataset"
)


# ==================================================
# 5. CALCULATE MOTION PER TRACK
# ==================================================

motion_columns = [
    "velocity_x_px_s",
    "velocity_y_px_s",
    "contact_velocity_x_px_s",
    "contact_velocity_y_px_s",
    "height_growth_px_s",
    "log_height_growth_s",
    "area_growth_px2_s",
]

for column in motion_columns:
    df[column] = np.nan

df["previous_frame"] = (
    df.groupby("track_id")["frame"].shift(1)
)

df["frame_gap"] = (
    df["frame"] - df["previous_frame"]
)

df["motion_valid"] = False

valid_pairs = 0

for track_id, indices in df.groupby(
    "track_id",
    sort=False,
).groups.items():

    track_indices = list(indices)

    for i in range(1, len(track_indices)):

        previous_index = track_indices[i - 1]
        current_index = track_indices[i]

        previous_row = df.loc[previous_index]
        current_row = df.loc[current_index]

        # Do not calculate frame-to-frame motion
        # across missing observations.
        if current_row["frame_gap"] != 1:
            continue

        previous_features = {
            key: float(previous_row[key])
            for key in [
                "center_x",
                "center_y",
                "width",
                "height",
                "area",
                "contact_x",
                "contact_y",
            ]
        }

        current_features = {
            key: float(current_row[key])
            for key in previous_features
        }

        motion = calculate_image_motion(
            previous_features,
            current_features,
            frame_dt,
        )

        for key, value in motion.items():
            df.at[current_index, key] = value

        df.at[current_index, "motion_valid"] = True

        valid_pairs += 1


# ==================================================
# 6. MOTION MAGNITUDE AND DIAGNOSTICS
# ==================================================

df["image_speed_px_s"] = np.hypot(
    df["velocity_x_px_s"],
    df["velocity_y_px_s"],
)

# Diagnostic-only screen-normalized displacement.
# This is not physical velocity.
df["normalized_frame_displacement"] = (
    np.hypot(
        df["velocity_x_px_s"] / fps / width,
        df["velocity_y_px_s"] / fps / height,
    )
)

# Flag large apparent motion for inspection.
# This is an arbitrary diagnostic threshold,
# not a validated tracker-error detector.
JUMP_THRESHOLD = 0.10

df["large_motion_jump"] = (
    df["motion_valid"]
    & (
        df["normalized_frame_displacement"]
        > JUMP_THRESHOLD
    )
)


# ==================================================
# 7. SAVE RESULTS
# ==================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n--- MOTION DATASET ---")
print(f"Consecutive valid pairs: {valid_pairs}")

print(
    "Observations without consecutive motion: "
    f"{len(df) - valid_pairs}"
)

print(
    "Large image-motion jumps: "
    f"{int(df['large_motion_jump'].sum())}"
)


# ==================================================
# 8. GLOBAL MOTION STATISTICS
# ==================================================

valid = df[df["motion_valid"]].copy()

print("\n--- IMAGE MOTION STATISTICS ---")

metrics = [
    "velocity_x_px_s",
    "velocity_y_px_s",
    "image_speed_px_s",
    "height_growth_px_s",
    "log_height_growth_s",
]

if not valid.empty:
    print(
        valid[metrics]
        .describe(percentiles=[0.50, 0.90, 0.95])
        .round(3)
        .to_string()
    )


# ==================================================
# 9. SELECTED TRACKS
# ==================================================

print("\n--- SELECTED TRACK ANALYSIS ---")

for track_id in SELECTED_TRACKS:

    track = df[
        df["track_id"] == track_id
    ]

    if track.empty:
        print(f"\nTrack {track_id}: not found")
        continue

    motion = track[
        track["motion_valid"]
    ]

    print(f"\nTrack {track_id}")

    print(f"Observations: {len(track)}")
    print(f"Valid motion pairs: {len(motion)}")

    print(
        "Frame range: "
        f"{int(track['frame'].min())} - "
        f"{int(track['frame'].max())}"
    )

    print(
        "Median box height: "
        f"{track['height'].median():.2f} px"
    )

    if not motion.empty:

        print(
            "Median horizontal velocity: "
            f"{motion['velocity_x_px_s'].median():.2f} px/s"
        )

        print(
            "Median vertical velocity: "
            f"{motion['velocity_y_px_s'].median():.2f} px/s"
        )

        print(
            "Median log-height growth: "
            f"{motion['log_height_growth_s'].median():.4f} 1/s"
        )

        print(
            "95th percentile image speed: "
            f"{motion['image_speed_px_s'].quantile(0.95):.2f} px/s"
        )


# ==================================================
# 10. FINAL OUTPUT
# ==================================================

print("\n--- OUTPUT ---")
print(OUTPUT_PATH)

print("\nImage-space motion analysis complete!")
