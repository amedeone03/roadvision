
import argparse
import pandas as pd


# --------------------------------------------------
# 1. Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Analyze RoadVision tracking results"
)

parser.add_argument(
    "--csv",
    type=str,
    default="runs/tracking_results.csv",
    help="Path to the tracking CSV file"
)

parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="Frame rate of the input video"
)

parser.add_argument(
    "--total-frames",
    type=int,
    default=1728,
    help="Total number of frames in the input video"
)

args = parser.parse_args()

if args.fps <= 0:
    parser.error("--fps must be greater than zero")

if args.total_frames <= 0:
    parser.error("--total-frames must be greater than zero")


# --------------------------------------------------
# 2. Load tracking data
# --------------------------------------------------

df = pd.read_csv(args.csv)

print(f"\nAnalyzing: {args.csv}")


# --------------------------------------------------
# 3. Basic statistics
# --------------------------------------------------

print("\n--- BASIC STATISTICS ---")

print("Total detection rows:", len(df))
print("Unique track IDs:", df["track_id"].nunique())

print("\nDetections by class:")
print(df["class"].value_counts())


# --------------------------------------------------
# 4. Group observations by track
# --------------------------------------------------

track_stats = df.groupby("track_id").agg(
    first_frame=("frame", "min"),
    last_frame=("frame", "max"),
    observed_frames=("frame", "nunique"),
)


# --------------------------------------------------
# 5. Calculate track duration
# --------------------------------------------------

track_stats["duration_seconds"] = (
    track_stats["last_frame"]
    - track_stats["first_frame"]
    + 1
) / args.fps


# --------------------------------------------------
# 6. Calculate missing frames
# --------------------------------------------------

track_stats["missing_frames"] = (
    track_stats["last_frame"]
    - track_stats["first_frame"]
    + 1
    - track_stats["observed_frames"]
)


# --------------------------------------------------
# 7. Track statistics
# --------------------------------------------------

print("\n--- TRACK STATISTICS ---")

print(track_stats.describe())

print("\n10 longest tracks:")

print(
    track_stats.sort_values(
        "duration_seconds",
        ascending=False
    ).head(10)
)


# --------------------------------------------------
# 8. Tracking coverage
# --------------------------------------------------

frames_with_tracks = df["frame"].nunique()

coverage = frames_with_tracks / args.total_frames


# --------------------------------------------------
# 9. Short-lived tracks
# --------------------------------------------------

short_tracks = (
    track_stats["observed_frames"] < 5
).sum()

total_tracks = len(track_stats)

short_track_percentage = (
    short_tracks / total_tracks
    if total_tracks > 0
    else 0
)


# --------------------------------------------------
# 10. Additional metrics
# --------------------------------------------------

print("\n--- ADDITIONAL METRICS ---")

print(f"Frames with tracked objects: {frames_with_tracks}")
print(f"Frame coverage: {coverage:.2%}")

print(f"Tracks with fewer than 5 observations: {short_tracks}")
print(f"Short track percentage: {short_track_percentage:.2%}")

if total_tracks > 0:
    print(
        f"Average observed frames per track: "
        f"{track_stats['observed_frames'].mean():.2f}"
    )

    print(
        f"Median track span: "
        f"{track_stats['duration_seconds'].median():.3f} s"
    )

    print(
        f"Mean track span: "
        f"{track_stats['duration_seconds'].mean():.3f} s"
    )

    print(
        f"Longest track span: "
        f"{track_stats['duration_seconds'].max():.3f} s"
    )

print("\nAnalysis complete!")
