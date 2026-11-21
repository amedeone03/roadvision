
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

INPUT = Path("runs/phase6/final/phase6_results.csv")
VIDEO = Path("data/videos/test_drive.mp4")

OUTPUT = Path("runs/phase7/tracking")

SHORT_TRACK_THRESHOLD = 5
LONG_GAP_THRESHOLD = 3

# Heuristic candidate search, not an ID-switch metric.
MAX_REAPPEARANCE_GAP = 10
MAX_NORMALIZED_CENTER_DISTANCE = 1.5

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]

REQUIRED_COLUMNS = [
    "frame",
    "track_id",
    "class_name",
    *BOX_COLUMNS,
]


# ==================================================
# VIDEO METADATA
# ==================================================

def get_video_metadata():
    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        cap.release()

    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("Invalid video metadata")

    return fps, total_frames


# ==================================================
# LOAD AND VALIDATE TRACKING DATA
# ==================================================

def load_tracking_data():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Tracking data contains missing required values"
        )

    for column in ["frame", "track_id"]:
        values = pd.to_numeric(
            df[column], errors="raise"
        )

        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {column}")

        if not (values == values.astype(int)).all():
            raise ValueError(
                f"{column} must contain integers"
            )

        df[column] = values.astype(int)

    for column in BOX_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    if not np.isfinite(df[BOX_COLUMNS].to_numpy()).all():
        raise ValueError("Nonfinite bounding-box coordinates")

    if (
        (df["x2"] <= df["x1"]).any()
        or (df["y2"] <= df["y1"]).any()
    ):
        raise ValueError("Invalid bounding boxes")

    if (df["frame"] < 1).any():
        raise ValueError("Frame numbers must start at 1")

    if df.duplicated(["frame", "track_id"]).any():
        raise ValueError(
            "Duplicate track IDs within the same frame"
        )

    return df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)


# ==================================================
# TRACK CONTINUITY
# ==================================================

def analyze_track(track, fps):
    frames = np.sort(
        track["frame"].to_numpy(dtype=int)
    )

    differences = np.diff(frames)

    # A difference of 1 means consecutive frames.
    gap_sizes = differences[differences > 1] - 1

    start_frame = int(frames[0])
    end_frame = int(frames[-1])

    observed_frames = len(frames)
    span_frames = end_frame - start_frame + 1

    missing_frames = int(gap_sizes.sum())
    gap_count = len(gap_sizes)

    # Number of contiguous segments within this track.
    segments = gap_count + 1

    # Compute the longest consecutive run.
    run_lengths = []
    current_run = 1

    for difference in differences:
        if difference == 1:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1

    run_lengths.append(current_run)

    classes = track["class_name"].astype(str)
    class_changes = int(
        (classes != classes.shift()).iloc[1:].sum()
    )

    first = track.iloc[0]
    last = track.iloc[-1]

    return {
        "track_id": int(first["track_id"]),
        "class_name": classes.mode().iloc[0],
        "distinct_classes": int(classes.nunique()),
        "class_changes": class_changes,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "observed_frames": observed_frames,
        "span_frames": span_frames,
        "span_seconds": span_frames / fps,
        "continuity_ratio": (
            observed_frames / span_frames
        ),
        "gap_count": gap_count,
        "missing_frames": missing_frames,
        "max_gap_frames": (
            int(gap_sizes.max())
            if len(gap_sizes) else 0
        ),
        "segments": segments,
        "longest_run": max(run_lengths),
        "short_lived": (
            observed_frames <= SHORT_TRACK_THRESHOLD
        ),
        "first_x1": float(first["x1"]),
        "first_y1": float(first["y1"]),
        "first_x2": float(first["x2"]),
        "first_y2": float(first["y2"]),
        "last_x1": float(last["x1"]),
        "last_y1": float(last["y1"]),
        "last_x2": float(last["x2"]),
        "last_y2": float(last["y2"]),
    }


# ==================================================
# INTERNAL GAP REPORT
# ==================================================

def extract_gaps(df):
    rows = []

    for track_id, track in df.groupby("track_id"):
        track = track.sort_values("frame")

        frames = track["frame"].to_numpy(dtype=int)

        for previous, current in zip(
            frames[:-1], frames[1:]
        ):
            missing = int(current - previous - 1)

            if missing <= 0:
                continue

            rows.append({
                "track_id": int(track_id),
                "last_seen_frame": int(previous),
                "next_seen_frame": int(current),
                "missing_frames": missing,
                "long_gap": (
                    missing >= LONG_GAP_THRESHOLD
                ),
            })

    return pd.DataFrame(
        rows,
        columns=[
            "track_id",
            "last_seen_frame",
            "next_seen_frame",
            "missing_frames",
            "long_gap",
        ],
    )


# ==================================================
# POTENTIAL TRACK FRAGMENTATION
# ==================================================

def box_center_and_size(row, prefix):
    x1 = row[f"{prefix}_x1"]
    y1 = row[f"{prefix}_y1"]
    x2 = row[f"{prefix}_x2"]
    y2 = row[f"{prefix}_y2"]

    return (
        np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        ]),
        max(x2 - x1, 1.0),
        max(y2 - y1, 1.0),
    )


def find_fragmentation_candidates(tracks):
    """
    Find pairs of different track IDs where one ends
    shortly before another starts in a nearby position.

    These are only VISUAL REVIEW CANDIDATES.
    They do not prove an ID switch or fragmentation.
    """
    rows = []

    for previous in tracks.itertuples(index=False):
        # Convert namedtuple to dictionary-like Series.
        previous = pd.Series(previous._asdict())

        for following in tracks.itertuples(index=False):
            following = pd.Series(following._asdict())

            if (
                previous["track_id"]
                == following["track_id"]
            ):
                continue

            if (
                previous["class_name"]
                != following["class_name"]
            ):
                continue

            frame_difference = int(
                following["start_frame"]
                - previous["end_frame"]
            )

            # Require nonoverlapping track lifetimes.
            if not (
                1 <= frame_difference
                <= MAX_REAPPEARANCE_GAP + 1
            ):
                continue

            center_a, width_a, height_a = (
                box_center_and_size(
                    previous, "last"
                )
            )

            center_b, width_b, height_b = (
                box_center_and_size(
                    following, "first"
                )
            )

            scale = max(
                (width_a + width_b) / 2,
                (height_a + height_b) / 2,
                1.0,
            )

            distance = float(
                np.linalg.norm(center_b - center_a)
                / scale
            )

            if (
                distance
                > MAX_NORMALIZED_CENTER_DISTANCE
            ):
                continue

            rows.append({
                "previous_track_id": int(
                    previous["track_id"]
                ),
                "following_track_id": int(
                    following["track_id"]
                ),
                "previous_end": int(
                    previous["end_frame"]
                ),
                "following_start": int(
                    following["start_frame"]
                ),
                "intervening_frames": (
                    frame_difference - 1
                ),
                "normalized_center_distance": distance,
                "class_name": previous["class_name"],
            })

    candidates = pd.DataFrame(
        rows,
        columns=[
            "previous_track_id",
            "following_track_id",
            "previous_end",
            "following_start",
            "intervening_frames",
            "normalized_center_distance",
            "class_name",
        ],
    )

    return candidates.sort_values(
        [
            "normalized_center_distance",
            "intervening_frames",
        ]
    ).reset_index(drop=True)


# ==================================================
# MAIN
# ==================================================

def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fps, total_frames = get_video_metadata()
    df = load_tracking_data()

    if df.empty:
        raise ValueError("No tracking observations")

    if df["frame"].max() > total_frames:
        raise ValueError(
            "Tracking data exceeds video frame count"
        )

    # Analyze each unique track.
    track_rows = [
        analyze_track(track, fps)
        for _, track in df.groupby("track_id")
    ]

    tracks = pd.DataFrame(track_rows)

    gaps = extract_gaps(df)

    candidates = find_fragmentation_candidates(
        tracks
    )

    # ------------------------------------------
    # Export detailed reports
    # ------------------------------------------

    tracks.to_csv(
        OUTPUT / "track_statistics.csv",
        index=False,
    )

    gaps.to_csv(
        OUTPUT / "track_gaps.csv",
        index=False,
    )

    candidates.to_csv(
        OUTPUT / "fragmentation_candidates.csv",
        index=False,
    )

    # ------------------------------------------
    # Global summary
    # ------------------------------------------

    summary = {
        "video_frames": total_frames,
        "fps": fps,
        "tracked_observations": len(df),
        "unique_tracks": len(tracks),
        "frames_with_tracks": int(
            df["frame"].nunique()
        ),
        "mean_track_observations": float(
            tracks["observed_frames"].mean()
        ),
        "median_track_observations": float(
            tracks["observed_frames"].median()
        ),
        "mean_continuity_ratio": float(
            tracks["continuity_ratio"].mean()
        ),
        "median_continuity_ratio": float(
            tracks["continuity_ratio"].median()
        ),
        "tracks_with_internal_gaps": int(
            (tracks["gap_count"] > 0).sum()
        ),
        "total_internal_gaps": len(gaps),
        "total_missing_frames_within_spans": int(
            tracks["missing_frames"].sum()
        ),
        "short_lived_tracks": int(
            tracks["short_lived"].sum()
        ),
        "tracks_with_class_changes": int(
            (tracks["class_changes"] > 0).sum()
        ),
        "fragmentation_candidates": len(
            candidates
        ),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUTPUT / "summary.csv",
        index=False,
    )

    # ------------------------------------------
    # Print summary
    # ------------------------------------------

    print("\n--- PHASE 7.4 TRACKING CONSISTENCY ---")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n--- TRACK LENGTH DISTRIBUTION ---")

    bins = [0, 5, 15, 30, 60, 120, np.inf]
    labels = [
        "1-5",
        "6-15",
        "16-30",
        "31-60",
        "61-120",
        ">120",
    ]

    distribution = pd.cut(
        tracks["observed_frames"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).value_counts().sort_index()

    print(distribution.to_string())

    print("\n--- TRACKS WITH MOST MISSING FRAMES ---")

    print(
        tracks.sort_values(
            "missing_frames",
            ascending=False,
        )[
            [
                "track_id",
                "observed_frames",
                "span_frames",
                "continuity_ratio",
                "gap_count",
                "missing_frames",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\n--- POSSIBLE FRAGMENTATION ---")

    if candidates.empty:
        print("No candidates under current heuristic.")
    else:
        print(
            candidates.head(10)
            .round(3)
            .to_string(index=False)
        )

    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    main()
