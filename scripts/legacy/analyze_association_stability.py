
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_PATH = Path("runs/integrated_restored.csv")

OUTPUT_DIR = Path("runs")
TRACK_SUMMARY_PATH = (
    OUTPUT_DIR / "association_track_summary.csv"
)
TRANSITIONS_PATH = (
    OUTPUT_DIR / "association_transitions.csv"
)

VALID_STATUSES = {
    "IN_LANE",
    "OUT_LANE",
    "UNKNOWN",
}


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def longest_consecutive_run(track, target_status):
    """
    Find the longest run of target_status.

    A run continues only when:
      - status matches target_status
      - frame numbers are consecutive

    Returns the number of frames in the longest run.
    """

    longest = 0
    current = 0
    previous_frame = None

    for row in track.itertuples(index=False):

        frame = int(row.frame)
        status = row.lane_status

        if (
            status == target_status
            and (
                previous_frame is None
                or frame == previous_frame + 1
            )
        ):
            current += 1

        elif status == target_status:
            current = 1

        else:
            current = 0

        longest = max(longest, current)
        previous_frame = frame

    return longest


def classify_transition(previous, current):
    """
    Classify a transition between two consecutive
    observations of the same track.
    """

    if previous == current:
        return "UNCHANGED"

    definite = {"IN_LANE", "OUT_LANE"}

    if previous in definite and current in definite:
        return "DIRECT_LANE_FLIP"

    if previous == "UNKNOWN" and current in definite:
        return "UNKNOWN_TO_DEFINITE"

    if previous in definite and current == "UNKNOWN":
        return "DEFINITE_TO_UNKNOWN"

    return "OTHER"


# --------------------------------------------------
# 3. Load and validate data
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Missing input file: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "frame",
    "track_id",
    "class_name",
    "lane_status",
    "lane_reason",
    "lane_valid",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if df.empty:
    raise RuntimeError("Input CSV contains no observations")

unexpected = set(df["lane_status"].dropna()) - VALID_STATUSES

if unexpected:
    raise ValueError(
        f"Unexpected lane statuses: {unexpected}"
    )

if df[list(required_columns)].isna().any().any():
    raise ValueError("Required columns contain missing values")

# Each track should have at most one observation per frame.
duplicate_rows = df.duplicated(
    subset=["track_id", "frame"]
)

if duplicate_rows.any():
    raise ValueError(
        "Duplicate (track_id, frame) observations found"
    )

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n--- ROADVISION ASSOCIATION STABILITY ---")
print(f"Input: {INPUT_PATH}")
print(f"Observations: {len(df)}")
print(f"Unique tracks: {df['track_id'].nunique()}")


# --------------------------------------------------
# 4. Identify consecutive-frame transitions
# --------------------------------------------------

df["previous_frame"] = (
    df.groupby("track_id")["frame"].shift(1)
)

df["previous_status"] = (
    df.groupby("track_id")["lane_status"].shift(1)
)

df["previous_reason"] = (
    df.groupby("track_id")["lane_reason"].shift(1)
)

df["previous_lane_valid"] = (
    df.groupby("track_id")["lane_valid"].shift(1)
)

df["frame_gap"] = df["frame"] - df["previous_frame"]

# Compare observations only if exactly one frame apart.
consecutive = df[
    df["frame_gap"] == 1
].copy()

consecutive["transition_type"] = [
    classify_transition(previous, current)
    for previous, current in zip(
        consecutive["previous_status"],
        consecutive["lane_status"],
    )
]

# Keep status changes only.
transitions = consecutive[
    consecutive["transition_type"] != "UNCHANGED"
].copy()

transition_columns = [
    "track_id",
    "frame",
    "class_name",
    "previous_status",
    "lane_status",
    "previous_reason",
    "lane_reason",
    "previous_lane_valid",
    "lane_valid",
    "transition_type",
]

transitions[transition_columns].to_csv(
    TRANSITIONS_PATH,
    index=False,
)


# --------------------------------------------------
# 5. Transition statistics
# --------------------------------------------------

print("\n--- CONSECUTIVE-FRAME TRANSITIONS ---")

print(f"Consecutive observation pairs: {len(consecutive)}")
print(f"Status changes: {len(transitions)}")

print("\nTransition types:")

transition_counts = (
    transitions["transition_type"].value_counts()
)

print(transition_counts.to_string())

direct_flips = transitions[
    transitions["transition_type"] == "DIRECT_LANE_FLIP"
]

print(f"\nDirect IN_LANE / OUT_LANE flips: {len(direct_flips)}")

if not direct_flips.empty:
    print("\nFirst 15 direct flips:")

    print(
        direct_flips[
            [
                "track_id",
                "frame",
                "class_name",
                "previous_status",
                "lane_status",
            ]
        ].head(15).to_string(index=False)
    )


# --------------------------------------------------
# 6. Per-track statistics
# --------------------------------------------------

track_summaries = []

for track_id, track in df.groupby("track_id"):

    track = track.sort_values("frame")

    total = len(track)

    in_count = int(
        (track["lane_status"] == "IN_LANE").sum()
    )

    out_count = int(
        (track["lane_status"] == "OUT_LANE").sum()
    )

    unknown_count = int(
        (track["lane_status"] == "UNKNOWN").sum()
    )

    track_transitions = transitions[
        transitions["track_id"] == track_id
    ]

    track_summaries.append({
        "track_id": int(track_id),
        "class_name": track["class_name"].mode().iloc[0],
        "first_frame": int(track["frame"].min()),
        "last_frame": int(track["frame"].max()),
        "observations": total,
        "in_lane_count": in_count,
        "out_lane_count": out_count,
        "unknown_count": unknown_count,
        "in_lane_pct": round(100 * in_count / total, 2),
        "out_lane_pct": round(100 * out_count / total, 2),
        "unknown_pct": round(
            100 * unknown_count / total, 2
        ),
        "longest_in_lane_run": longest_consecutive_run(
            track, "IN_LANE"
        ),
        "longest_out_lane_run": longest_consecutive_run(
            track, "OUT_LANE"
        ),
        "direct_lane_flips": int(
            (
                track_transitions["transition_type"]
                == "DIRECT_LANE_FLIP"
            ).sum()
        ),
        "uncertainty_transitions": int(
            track_transitions["transition_type"].isin(
                [
                    "UNKNOWN_TO_DEFINITE",
                    "DEFINITE_TO_UNKNOWN",
                ]
            ).sum()
        ),
    })

summary = pd.DataFrame(track_summaries)

summary = summary.sort_values(
    ["direct_lane_flips", "observations"],
    ascending=[False, False],
)

summary.to_csv(
    TRACK_SUMMARY_PATH,
    index=False,
)


# --------------------------------------------------
# 7. Report most unstable tracks
# --------------------------------------------------

print("\n--- TRACKS WITH MOST DIRECT LANE FLIPS ---")

unstable = summary[
    summary["direct_lane_flips"] > 0
]

if unstable.empty:
    print("No direct IN_LANE / OUT_LANE flips detected.")
else:
    print(
        unstable[
            [
                "track_id",
                "class_name",
                "observations",
                "direct_lane_flips",
                "uncertainty_transitions",
                "longest_in_lane_run",
            ]
        ].head(10).to_string(index=False)
    )


# --------------------------------------------------
# 8. Report longest IN_LANE sequences
# --------------------------------------------------

print("\n--- LONGEST CONTINUOUS IN_LANE RUNS ---")

longest_runs = summary.sort_values(
    "longest_in_lane_run",
    ascending=False,
)

print(
    longest_runs[
        [
            "track_id",
            "class_name",
            "observations",
            "in_lane_count",
            "longest_in_lane_run",
        ]
    ].head(10).to_string(index=False)
)


# --------------------------------------------------
# 9. Global summary
# --------------------------------------------------

print("\n--- GLOBAL SUMMARY ---")

print(
    "Tracks with at least one IN_LANE observation: "
    f"{(summary['in_lane_count'] > 0).sum()}"
)

print(
    "Tracks with at least one OUT_LANE observation: "
    f"{(summary['out_lane_count'] > 0).sum()}"
)

print(
    "Tracks with direct lane flips: "
    f"{(summary['direct_lane_flips'] > 0).sum()}"
)

if len(consecutive) > 0:
    flip_rate = len(direct_flips) / len(consecutive)

    print(
        "Direct flip rate per consecutive pair: "
        f"{flip_rate:.2%}"
    )

print(f"\nTrack summary saved to: {TRACK_SUMMARY_PATH}")
print(f"Transitions saved to: {TRANSITIONS_PATH}")

print("\nAnalysis complete!")
