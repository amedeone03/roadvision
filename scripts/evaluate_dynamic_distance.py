
from pathlib import Path

import numpy as np
import pandas as pd


TRACKING_CSV = Path("runs/integrated_final.csv")

VP_CSV = Path(
    "runs/motion/full_video_vp/temporal_vp_results.csv"
)

OUTPUT_CSV = Path(
    "runs/motion/dynamic_distance_results.csv"
)

METHOD = "EDGE_MASKED"

# Hypothetical parameters.
CAMERA_HEIGHT_M = 1.5
FOCAL_LENGTH_PX = 1000.0

MIN_HORIZON_SEPARATION_PX = 30.0

# Evaluate uncertainty induced by VP perturbations.
VP_UNCERTAINTY_PX = 10.0

SELECTED_TRACKS = [2, 8, 63, 66, 125, 127]


def estimate_distance(contact_y, vp_y):
    """
    Simplified ground-plane distance approximation.

    Returns NaN when the contact point is too close
    to or above the estimated horizon.
    """

    if pd.isna(contact_y) or pd.isna(vp_y):
        return np.nan

    separation = float(contact_y - vp_y)

    if separation <= MIN_HORIZON_SEPARATION_PX:
        return np.nan

    return (
        CAMERA_HEIGHT_M
        * FOCAL_LENGTH_PX
        / separation
    )


def main():

    tracks = pd.read_csv(TRACKING_CSV)
    vp = pd.read_csv(VP_CSV)

    required_tracks = {
        "frame", "track_id", "contact_y",
        "lane_status",
    }

    required_vp = {
        "frame", "method", "filtered_vp_y",
    }

    missing_tracks = required_tracks - set(tracks.columns)
    missing_vp = required_vp - set(vp.columns)

    if missing_tracks or missing_vp:
        raise ValueError(
            f"Missing tracking columns: {missing_tracks}; "
            f"missing VP columns: {missing_vp}"
        )

    vp = vp[
        vp["method"] == METHOD
    ][
        ["frame", "filtered_vp_y"]
    ].copy()

    if vp["frame"].duplicated().any():
        raise ValueError(
            "Duplicate VP estimates for the same frame"
        )

    df = tracks.merge(
        vp,
        on="frame",
        how="left",
        validate="many_to_one",
    )

    # ==========================================
    # DISTANCE SCENARIOS
    # ==========================================

    df["distance_dynamic_m"] = [
        estimate_distance(contact, horizon)
        for contact, horizon in zip(
            df["contact_y"],
            df["filtered_vp_y"],
        )
    ]

    df["distance_vp_minus_10_m"] = [
        estimate_distance(contact, horizon - VP_UNCERTAINTY_PX)
        if pd.notna(horizon) else np.nan
        for contact, horizon in zip(
            df["contact_y"],
            df["filtered_vp_y"],
        )
    ]

    df["distance_vp_plus_10_m"] = [
        estimate_distance(contact, horizon + VP_UNCERTAINTY_PX)
        if pd.notna(horizon) else np.nan
        for contact, horizon in zip(
            df["contact_y"],
            df["filtered_vp_y"],
        )
    ]

    scenario_columns = [
        "distance_vp_minus_10_m",
        "distance_dynamic_m",
        "distance_vp_plus_10_m",
    ]

    df["distance_sensitivity_range_m"] = (
        df[scenario_columns].max(axis=1)
        - df[scenario_columns].min(axis=1)
    )

    # Only evaluate sensitivity when all scenarios
    # satisfy the same geometric validity condition.
    all_valid = df[scenario_columns].notna().all(axis=1)

    df.loc[
        ~all_valid,
        "distance_sensitivity_range_m",
    ] = np.nan

    df["relative_distance_sensitivity"] = (
        df["distance_sensitivity_range_m"]
        / df["distance_dynamic_m"]
    )

    # Diagnostic threshold, not a safety guarantee.
    df["high_distance_sensitivity"] = (
        df["relative_distance_sensitivity"] > 0.20
    )

    # ==========================================
    # RESULTS
    # ==========================================

    print("\n--- DYNAMIC VP DISTANCE ESTIMATION ---")
    print(f"Observations: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")

    valid = int(
        df["distance_dynamic_m"].notna().sum()
    )

    print(
        f"Valid dynamic distances: "
        f"{valid}/{len(df)} "
        f"({100 * valid / len(df):.2f}%)"
    )

    print(
        "High distance sensitivity:",
        int(df["high_distance_sensitivity"].sum()),
    )

    print("\n--- GLOBAL DISTANCE STATISTICS ---")

    print(
        df["distance_dynamic_m"]
        .describe(
            percentiles=[0.5, 0.9, 0.95]
        )
        .round(2)
        .to_string()
    )

    # ==========================================
    # SELECTED TRACKS
    # ==========================================

    print("\n--- SELECTED TRACKS ---")

    for track_id in SELECTED_TRACKS:

        subset = df[
            df["track_id"] == track_id
        ]

        if subset.empty:
            continue

        distances = subset["distance_dynamic_m"]

        print(f"\nTrack {track_id}")
        print(f"Observations: {len(subset)}")

        print(
            f"Valid distances: "
            f"{distances.notna().sum()}"
        )

        if distances.notna().any():
            print(
                "Median distance: "
                f"{distances.median():.2f} m"
            )
            print(
                "P95 distance: "
                f"{distances.quantile(0.95):.2f} m"
            )

        print(
            "High-sensitivity observations:",
            int(
                subset[
                    "high_distance_sensitivity"
                ].sum()
            ),
        )

    # ==========================================
    # SAVE
    # ==========================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_CSV}")
    print("\nDynamic distance experiment complete!")


if __name__ == "__main__":
    main()
