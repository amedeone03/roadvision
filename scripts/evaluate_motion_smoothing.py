
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------
# 1. Configuration
# -----------------------------------------------

INPUT_PATH = Path(
    "runs/motion/image_motion_results.csv"
)

OUTPUT_DIR = Path("runs/motion")
OUTPUT_PATH = OUTPUT_DIR / "motion_smoothing_results.csv"

WINDOWS = [5, 9]
SELECTED_TRACKS = [2, 8, 63, 66, 125]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------
# 2. Load and validate data
# -----------------------------------------------

df = pd.read_csv(INPUT_PATH)

required = {
    "frame",
    "track_id",
    "time_seconds",
    "height",
    "log_height_growth_s",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

if df.empty:
    raise RuntimeError("Empty motion dataset")

if df.duplicated(["track_id", "frame"]).any():
    raise ValueError("Duplicate track/frame pairs")

if (
    (df["height"] <= 0).any()
    or not np.isfinite(df["height"]).all()
):
    raise ValueError("Invalid bounding-box heights")

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)

# Logarithmic apparent size.
df["log_height"] = np.log(df["height"])


# -----------------------------------------------
# 3. Separate consecutive tracking segments
# -----------------------------------------------

frame_gap = (
    df.groupby("track_id")["frame"].diff()
)

# Start a new segment after every tracking gap.
new_segment = (
    frame_gap.isna()
    | frame_gap.ne(1)
)

df["segment_id"] = (
    new_segment.groupby(df["track_id"])
    .cumsum()
)

# -----------------------------------------------
# 4. Rolling slope estimator
# -----------------------------------------------

def rolling_log_slope(group, window):
    """
    Estimate d(log(height))/dt using linear
    regression over the last N observations.

    Uses no future observations.
    Returns NaN until N consecutive
    observations are available.
    """

    times = group["time_seconds"].to_numpy(
        dtype=float
    )

    log_heights = group["log_height"].to_numpy(
        dtype=float
    )

    result = np.full(
        len(group),
        np.nan,
        dtype=float,
    )

    for i in range(window - 1, len(group)):

        start = i - window + 1

        t = times[start:i + 1]
        y = log_heights[start:i + 1]

        # Center times to improve numerical stability.
        t = t - t.mean()

        denominator = np.sum(t ** 2)

        if denominator <= 0:
            continue

        slope = (
            np.sum(t * (y - y.mean()))
            / denominator
        )

        result[i] = slope

    return result


# -----------------------------------------------
# 5. Evaluate windows per track segment
# -----------------------------------------------

for window in WINDOWS:
    df[f"growth_lr_{window}"] = np.nan

for (_, _), group in df.groupby(
    ["track_id", "segment_id"],
    sort=False,
):

    for window in WINDOWS:

        values = rolling_log_slope(
            group,
            window,
        )

        df.loc[
            group.index,
            f"growth_lr_{window}"
        ] = values


# -----------------------------------------------
# 6. Export results
# -----------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n--- ROADVISION MOTION SMOOTHING ---")
print(f"Observations: {len(df)}")
print(f"Unique tracks: {df['track_id'].nunique()}")
print(f"Consecutive segments: {len(df.groupby(['track_id', 'segment_id']))}")


# -----------------------------------------------
# 7. Global comparison
# -----------------------------------------------

print("\n--- GLOBAL GROWTH COMPARISON ---")

methods = {
    "RAW": "log_height_growth_s",
    "LR_5": "growth_lr_5",
    "LR_9": "growth_lr_9",
}

for name, column in methods.items():

    values = df[column].dropna()

    if values.empty:
        continue

    print(f"\n{name}")
    print(f"Valid estimates: {len(values)}")
    print(f"Median: {values.median():.4f} 1/s")
    print(
        f"Median absolute growth: "
        f"{values.abs().median():.4f} 1/s"
    )
    print(
        f"95th percentile absolute growth: "
        f"{values.abs().quantile(0.95):.4f} 1/s"
    )


# -----------------------------------------------
# 8. Compare selected tracks
# -----------------------------------------------

print("\n--- SELECTED TRACKS ---")

for track_id in SELECTED_TRACKS:

    track = df[
        df["track_id"] == track_id
    ]

    if track.empty:
        continue

    print(f"\nTrack {track_id}")

    for name, column in methods.items():

        values = track[column].dropna()

        if values.empty:
            continue

        print(
            f"  {name}: "
            f"median={values.median():+.4f}, "
            f"p95_abs={values.abs().quantile(0.95):.4f}, "
            f"n={len(values)}"
        )


# -----------------------------------------------
# 9. Inspect known noisy frames
# -----------------------------------------------

print("\n--- TRACK 63: FRAMES 595-605 ---")

example = df[
    (df["track_id"] == 63)
    & df["frame"].between(595, 605)
]

print(
    example[
        [
            "frame",
            "height",
            "log_height_growth_s",
            "growth_lr_5",
            "growth_lr_9",
        ]
    ].round(3).to_string(index=False)
)

print("\n--- TRACK 63: FRAMES 694-701 ---")

example = df[
    (df["track_id"] == 63)
    & df["frame"].between(694, 701)
]

print(
    example[
        [
            "frame",
            "height",
            "log_height_growth_s",
            "growth_lr_5",
            "growth_lr_9",
        ]
    ].round(3).to_string(index=False)
)

print(f"\nSaved: {OUTPUT_PATH}")
print("\nExperiment complete!")
