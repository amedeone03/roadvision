
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_PATH = Path("runs/integrated_restored.csv")

OUTPUT_DIR = Path("runs/association_debug")

TRACK_IDS = [2, 8, 125]

# Specific frame intervals to inspect closely.
FRAME_WINDOWS = {
    8: (295, 330),
    125: (995, 1025),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "frame",
    "timestamp",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "contact_x",
    "contact_y",
    "lane_status",
    "lane_reason",
    "lane_valid",
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df.sort_values(["track_id", "frame"])

print("\n--- ROADVISION: PROBLEMATIC TRACK ANALYSIS ---")


# --------------------------------------------------
# 3. Analyze selected tracks
# --------------------------------------------------

for track_id in TRACK_IDS:

    track = df[df["track_id"] == track_id].copy()

    if track.empty:
        print(f"\nTrack {track_id}: not found")
        continue

    # Additional geometric information.
    track["bbox_width"] = track["x2"] - track["x1"]
    track["bbox_height"] = track["y2"] - track["y1"]

    # Previous observation of the same track.
    track["previous_frame"] = track["frame"].shift(1)
    track["previous_status"] = track["lane_status"].shift(1)

    # Only consecutive-frame comparisons.
    track["consecutive"] = (
        track["frame"] - track["previous_frame"] == 1
    )

    track["status_changed"] = (
        track["consecutive"]
        & track["lane_status"].ne(track["previous_status"])
    )

    # Direct IN_LANE <-> OUT_LANE changes.
    definite_statuses = {"IN_LANE", "OUT_LANE"}

    track["direct_flip"] = (
        track["status_changed"]
        & track["lane_status"].isin(definite_statuses)
        & track["previous_status"].isin(definite_statuses)
    )

    # ------------------------------------------
    # General track statistics
    # ------------------------------------------

    print(f"\n{'=' * 50}")
    print(f"TRACK ID: {track_id}")
    print(f"{'=' * 50}")

    print(f"Class: {track['class_name'].mode().iloc[0]}")
    print(f"Observations: {len(track)}")

    print(
        f"Frame range: "
        f"{track['frame'].min()} - {track['frame'].max()}"
    )

    print("\nLane classifications:")
    print(track["lane_status"].value_counts().to_string())

    print("\nClassification reasons:")
    print(track["lane_reason"].value_counts().to_string())

    print(
        f"\nConsecutive status changes: "
        f"{int(track['status_changed'].sum())}"
    )

    print(
        f"Direct lane flips: "
        f"{int(track['direct_flip'].sum())}"
    )

    print(
        f"Median bounding-box width: "
        f"{track['bbox_width'].median():.1f} px"
    )

    print(
        f"Median bounding-box height: "
        f"{track['bbox_height'].median():.1f} px"
    )

    # ------------------------------------------
    # Export complete track
    # ------------------------------------------

    output_path = (
        OUTPUT_DIR / f"track_{track_id}_observations.csv"
    )

    track.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")

    # ------------------------------------------
    # Inspect direct flips
    # ------------------------------------------

    flips = track[track["direct_flip"]]

    if not flips.empty:

        print("\nDirect lane flips:")

        columns = [
            "frame",
            "timestamp",
            "previous_status",
            "lane_status",
            "contact_x",
            "contact_y",
            "bbox_width",
            "bbox_height",
        ]

        print(flips[columns].to_string(index=False))

    # ------------------------------------------
    # Export selected frame interval
    # ------------------------------------------

    if track_id in FRAME_WINDOWS:

        start_frame, end_frame = FRAME_WINDOWS[track_id]

        window = track[
            track["frame"].between(
                start_frame,
                end_frame,
            )
        ].copy()

        window_path = (
            OUTPUT_DIR / f"track_{track_id}_window.csv"
        )

        window.to_csv(window_path, index=False)

        print(
            f"\nDiagnostic window: "
            f"frames {start_frame}-{end_frame}"
        )

        print(f"Saved: {window_path}")

        columns = [
            "frame",
            "lane_status",
            "lane_reason",
            "contact_x",
            "contact_y",
            "lane_valid",
        ]

        print("\nFrame-by-frame results:")
        print(window[columns].to_string(index=False))


print("\n--- ANALYSIS COMPLETE ---")
