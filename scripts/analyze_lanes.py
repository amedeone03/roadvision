
import argparse

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Analyze RoadVision lane detection stability"
)

parser.add_argument(
    "--csv",
    type=str,
    default="runs/lane_results.csv",
)

parser.add_argument(
    "--jump-threshold",
    type=float,
    default=100.0,
    help="Diagnostic boundary jump threshold in pixels",
)

args = parser.parse_args()


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(args.csv)

df = df.sort_values("frame").reset_index(drop=True)

print(f"\nAnalyzing: {args.csv}")

total_frames = len(df)

if total_frames == 0:
    raise RuntimeError("The lane CSV contains no frames")


# --------------------------------------------------
# 3. Basic statistics
# --------------------------------------------------

valid_frames = int(df["valid"].sum())
invalid_frames = total_frames - valid_frames

print("\n--- BASIC STATISTICS ---")
print(f"Total frames: {total_frames}")
print(f"Valid frames: {valid_frames}")
print(f"Invalid frames: {invalid_frames}")
print(f"Validation rate: {valid_frames / total_frames:.2%}")


# --------------------------------------------------
# 4. Missing boundary analysis
# --------------------------------------------------

missing_left = df["left_bottom_x"].isna()
missing_right = df["right_bottom_x"].isna()

print("\n--- MISSING BOUNDARIES ---")

print(f"Frames missing left boundary: {missing_left.sum()}")
print(f"Frames missing right boundary: {missing_right.sum()}")

print(
    f"Frames missing both boundaries: "
    f"{(missing_left & missing_right).sum()}"
)


# --------------------------------------------------
# 5. Lane width statistics
# --------------------------------------------------

print("\n--- LANE WIDTH STATISTICS ---")

valid_df = df[df["valid"]].copy()

print(
    valid_df[["bottom_width", "top_width"]].describe()
)


# --------------------------------------------------
# 6. Frame-to-frame boundary changes
# --------------------------------------------------

# Compare each boundary with the previous frame
# only when both estimates exist in consecutive
# video frames.

boundary_columns = [
    "left_bottom_x",
    "right_bottom_x",
    "left_top_x",
    "right_top_x",
]

print("\n--- BOUNDARY STABILITY ---")

for column in boundary_columns:

    previous = df[column].shift(1)

    # Exclude missing observations and frame gaps
    consecutive = df["frame"].diff().eq(1)

    jumps = (df[column] - previous).abs()

    jumps = jumps.where(
        consecutive
        & df[column].notna()
        & previous.notna()
    )

    large_jumps = jumps > args.jump_threshold

    print(f"\n{column}")
    print(f"Comparable frame pairs: {jumps.notna().sum()}")
    print(f"Median jump: {jumps.median():.2f} px")
    print(f"95th percentile jump: {jumps.quantile(0.95):.2f} px")
    print(f"Maximum jump: {jumps.max():.2f} px")
    print(f"Jumps > {args.jump_threshold:g} px: {large_jumps.sum()}")


# --------------------------------------------------
# 7. Consecutive invalid-frame intervals
# --------------------------------------------------

print("\n--- INVALID INTERVALS ---")

invalid = ~df["valid"]

# Start a new run when validity changes
# or when frame numbers are not consecutive.
new_run = (
    invalid.ne(invalid.shift(1))
    | df["frame"].diff().ne(1)
)

run_ids = new_run.cumsum()

invalid_runs = (
    df.loc[invalid]
    .groupby(run_ids[invalid])
    .agg(
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        length=("frame", "count"),
    )
)

print(f"Number of invalid intervals: {len(invalid_runs)}")

if not invalid_runs.empty:

    print(
        f"Longest invalid interval: "
        f"{invalid_runs['length'].max()} frames"
    )

    print("\n10 longest invalid intervals:")

    print(
        invalid_runs.sort_values(
            "length",
            ascending=False
        ).head(10).to_string(index=False)
    )


# --------------------------------------------------
# 8. Finish
# --------------------------------------------------

print("\nLane stability analysis complete!")
