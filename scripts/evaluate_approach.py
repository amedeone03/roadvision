
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.approach import classify_approach


# ==========================================
# CONFIGURATION
# ==========================================

MOTION_CSV = Path(
    "runs/motion/robust_motion_results.csv"
)

PHASE5_CSV = Path(
    "runs/phase5/phase5_results.csv"
)

OUTPUT_DIR = Path("runs/phase6")
OUTPUT_CSV = OUTPUT_DIR / "approach_results.csv"

THRESHOLDS = {
    "t010": 0.10,
    "t015": 0.15,
    "t020": 0.20,
}

BASELINE_COLUMN = "approach_t015"

SELECTED_TRACKS = [2, 8, 63, 66, 125, 127]

STATES = [
    "APPROACHING",
    "STABLE",
    "RECEDING",
    "UNKNOWN",
]


# ==========================================
# LOAD AND MERGE
# ==========================================

def load_data():

    motion = pd.read_csv(MOTION_CSV)
    phase5 = pd.read_csv(PHASE5_CSV)

    keys = ["frame", "track_id"]

    motion_required = set(keys + ["growth_robust_5"])
    phase5_required = set(
        keys + ["lane_status", "motion_quality"]
    )

    if not motion_required.issubset(motion.columns):
        raise ValueError(
            "Missing columns in motion CSV: "
            f"{sorted(motion_required - set(motion.columns))}"
        )

    if not phase5_required.issubset(phase5.columns):
        raise ValueError(
            "Missing columns in Phase 5 CSV: "
            f"{sorted(phase5_required - set(phase5.columns))}"
        )

    if motion.duplicated(keys).any():
        raise ValueError("Duplicate motion observations")

    if phase5.duplicated(keys).any():
        raise ValueError("Duplicate Phase 5 observations")

    # Merge only the growth estimate.
    # This avoids duplicate lane_status columns.
    df = phase5.merge(
        motion[keys + ["growth_robust_5"]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not df["_merge"].eq("both").all():
        raise ValueError(
            "Some Phase 5 observations have no "
            "matching motion row"
        )

    if len(df) != len(motion):
        raise ValueError(
            "Motion and Phase 5 row counts differ"
        )

    df = df.drop(columns="_merge")

    return df.sort_values(keys).reset_index(drop=True)


# ==========================================
# CLASSIFY APPROACH
# ==========================================

def add_approach_states(df):

    for name, threshold in THRESHOLDS.items():

        column = f"approach_{name}"

        df[column] = df[
            "growth_robust_5"
        ].apply(
            lambda growth: classify_approach(
                growth,
                threshold=threshold,
            )
        )

    return df


# ==========================================
# REPORT
# ==========================================

def print_distribution(df, column):

    counts = (
        df[column]
        .value_counts()
        .reindex(STATES, fill_value=0)
    )

    print(f"\n{column}")

    for state, count in counts.items():

        percentage = 100 * count / len(df)

        print(
            f"  {state:12s}: "
            f"{count:4d} ({percentage:5.2f}%)"
        )


def report_lane_association(df):

    print("\n--- MOTION STATE BY LANE ---")

    table = pd.crosstab(
        df["lane_status"],
        df[BASELINE_COLUMN],
    )

    table = table.reindex(
        columns=STATES,
        fill_value=0,
    )

    print(table.to_string())

    print("\n--- APPROACHING VEHICLES ---")

    approaching = df[
        df[BASELINE_COLUMN] == "APPROACHING"
    ]

    print("Observations:", len(approaching))

    print(
        "Unique tracks:",
        approaching["track_id"].nunique(),
    )

    print("\nLane distribution:")

    print(
        approaching["lane_status"]
        .value_counts()
        .to_string()
    )


def report_threshold_sensitivity(df):

    print("\n--- THRESHOLD SENSITIVITY ---")

    baseline = df[BASELINE_COLUMN]

    for column in [
        "approach_t010",
        "approach_t020",
    ]:

        changed = (
            df[column] != baseline
        ).sum()

        print(
            f"{column} vs {BASELINE_COLUMN}: "
            f"{changed} changed classifications "
            f"({100 * changed / len(df):.2f}%)"
        )


def report_selected_tracks(df):

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
            track[BASELINE_COLUMN]
            .value_counts()
            .reindex(STATES, fill_value=0)
            .to_string()
        )

        valid_growth = track[
            "growth_robust_5"
        ].dropna()

        if not valid_growth.empty:
            print(
                "Median robust growth: "
                f"{valid_growth.median():+.3f} /s"
            )


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_data()

    df = add_approach_states(df)

    print("\n--- PHASE 6 APPROACH EVALUATION ---")

    print("Observations:", len(df))
    print("Unique tracks:", df["track_id"].nunique())

    valid_growth = (
        df["growth_robust_5"]
        .notna()
        .sum()
    )

    print(
        f"Available robust growth estimates: "
        f"{valid_growth}/{len(df)}"
    )

    print("\n--- STATE DISTRIBUTIONS ---")

    for name in THRESHOLDS:
        print_distribution(
            df,
            f"approach_{name}",
        )

    report_lane_association(df)
    report_threshold_sensitivity(df)
    report_selected_tracks(df)

    # Verify missing estimates are UNKNOWN.
    missing_growth = (
        ~np.isfinite(
            pd.to_numeric(
                df["growth_robust_5"],
                errors="coerce",
            )
        )
    )

    assert (
        df.loc[
            missing_growth,
            BASELINE_COLUMN,
        ] == "UNKNOWN"
    ).all()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n--- OUTPUT ---")
    print(OUTPUT_CSV)
    print("\nApproach evaluation complete!")


if __name__ == "__main__":
    main()
