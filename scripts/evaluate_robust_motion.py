
from pathlib import Path

import numpy as np
import pandas as pd

from src.motion.robust_growth import (
    rolling_robust_growth,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

INPUT_PATH = Path(
    "runs/motion/motion_smoothing_results.csv"
)

OUTPUT_DIR = Path("runs/motion")

OUTPUT_PATH = (
    OUTPUT_DIR / "robust_motion_results.csv"
)

ROBUST_WINDOW = 5

SELECTED_TRACKS = [2, 8, 63, 66, 125]

METHODS = {
    "RAW": "log_height_growth_s",
    "LR_5": "growth_lr_5",
    "ROBUST_5": "growth_robust_5",
    "LR_9": "growth_lr_9",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

df = pd.read_csv(INPUT_PATH)

required = {
    "frame",
    "track_id",
    "time_seconds",
    "height",
    "segment_id",
    "log_height_growth_s",
    "growth_lr_5",
    "growth_lr_9",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

if df.empty:
    raise RuntimeError("Motion dataset is empty")

if df.duplicated(["track_id", "frame"]).any():
    raise ValueError("Duplicate track/frame observations")

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)

if (
    (df["height"] <= 0).any()
    or not np.isfinite(df["height"]).all()
):
    raise ValueError("Invalid bounding-box heights")


# ==================================================
# 3. COMPUTE ROBUST ROLLING GROWTH
# ==================================================

df["growth_robust_5"] = np.nan

for (_, _), group in df.groupby(
    ["track_id", "segment_id"],
    sort=False,
):

    estimates = rolling_robust_growth(
        group["time_seconds"].to_numpy(),
        group["height"].to_numpy(),
        window=ROBUST_WINDOW,
    )

    df.loc[
        group.index,
        "growth_robust_5"
    ] = estimates


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def growth_statistics(values):
    """Summarize a collection of growth estimates."""

    values = pd.Series(values).dropna()

    if values.empty:
        return {
            "n": 0,
            "median": np.nan,
            "median_abs": np.nan,
            "p95_abs": np.nan,
        }

    return {
        "n": len(values),
        "median": float(values.median()),
        "median_abs": float(values.abs().median()),
        "p95_abs": float(values.abs().quantile(0.95)),
    }


def consecutive_changes(data, column):
    """
    Absolute changes in a growth estimator between
    consecutive frames of the same track.

    Both current and previous estimates must exist.
    """

    ordered = data.sort_values(
        ["track_id", "frame"]
    ).copy()

    previous_frame = (
        ordered.groupby("track_id")["frame"].shift(1)
    )

    previous_value = (
        ordered.groupby("track_id")[column].shift(1)
    )

    consecutive = (
        ordered["frame"] - previous_frame == 1
    )

    valid = (
        consecutive
        & ordered[column].notna()
        & previous_value.notna()
    )

    changes = (
        ordered.loc[valid, column]
        - previous_value.loc[valid]
    ).abs()

    return changes


# ==================================================
# 5. GLOBAL COMPARISON
# ==================================================

print("\n--- ROADVISION ROBUST MOTION EXPERIMENT ---")

print(f"Observations: {len(df)}")
print(f"Unique tracks: {df['track_id'].nunique()}")

print("\n--- ESTIMATOR AVAILABILITY ---")

for name, column in METHODS.items():

    valid_count = df[column].notna().sum()

    print(
        f"{name}: {valid_count}/{len(df)} "
        f"({valid_count / len(df):.2%})"
    )


print("\n--- GLOBAL GROWTH STATISTICS ---")

for name, column in METHODS.items():

    stats = growth_statistics(df[column])

    print(
        f"{name}: "
        f"median={stats['median']:+.4f}, "
        f"median_abs={stats['median_abs']:.4f}, "
        f"p95_abs={stats['p95_abs']:.4f}, "
        f"n={stats['n']}"
    )


# ==================================================
# 6. FAIR COMPARISON: MATCHED OBSERVATIONS
# ==================================================

columns = list(METHODS.values())

matched = df.dropna(
    subset=columns
).copy()

print("\n--- MATCHED-OBSERVATION COMPARISON ---")
print(f"Matched observations: {len(matched)}")

for name, column in METHODS.items():

    stats = growth_statistics(matched[column])

    print(
        f"{name}: "
        f"median={stats['median']:+.4f}, "
        f"median_abs={stats['median_abs']:.4f}, "
        f"p95_abs={stats['p95_abs']:.4f}"
    )


# ==================================================
# 7. FAIR COMPARISON: CONSECUTIVE FRAME PAIRS
# ==================================================

print("\n--- MATCHED FRAME-TO-FRAME STABILITY ---")

# Preserve only pairs for which all four methods
# are available in both consecutive frames.
previous_frame = (
    df.groupby("track_id")["frame"].shift(1)
)

consecutive = (
    df["frame"] - previous_frame == 1
)

all_current_valid = df[columns].notna().all(axis=1)

all_previous_valid = (
    all_current_valid.groupby(df["track_id"])
    .shift(1)
    .fillna(False)
    .astype(bool)
)

pair_mask = (
    consecutive
    & all_current_valid
    & all_previous_valid
)

print(f"Matched consecutive pairs: {int(pair_mask.sum())}")

for name, column in METHODS.items():

    previous_value = (
        df.groupby("track_id")[column].shift(1)
    )

    changes = (
        df.loc[pair_mask, column]
        - previous_value.loc[pair_mask]
    ).abs()

    print(
        f"{name}: "
        f"median_change={changes.median():.4f}, "
        f"p95_change={changes.quantile(0.95):.4f}"
    )


# ==================================================
# 8. SELECTED TRACK COMPARISON
# ==================================================

print("\n--- SELECTED TRACKS ---")

for track_id in SELECTED_TRACKS:

    track = df[
        df["track_id"] == track_id
    ]

    if track.empty:
        continue

    print(f"\nTrack {track_id}")

    for name, column in METHODS.items():

        stats = growth_statistics(track[column])

        print(
            f"  {name}: "
            f"median={stats['median']:+.4f}, "
            f"p95_abs={stats['p95_abs']:.4f}, "
            f"n={stats['n']}"
        )


# ==================================================
# 9. KNOWN OUTLIER: TRACK 63
# ==================================================

print("\n--- TRACK 63: FRAMES 595–605 ---")

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
            "growth_robust_5",
            "growth_lr_9",
        ]
    ].round(3).to_string(index=False)
)


print("\n--- TRACK 63: FRAMES 694–701 ---")

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
            "growth_robust_5",
            "growth_lr_9",
        ]
    ].round(3).to_string(index=False)
)


# ==================================================
# 10. EXPORT RESULTS
# ==================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n--- OUTPUT ---")
print(OUTPUT_PATH)

print("\nRobust motion experiment complete!")
