
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "runs/motion/relative_motion_results.csv"
)

OUTPUT_PATH = Path(
    "runs/motion/motion_reliability_results.csv"
)

WINDOW = 5
MAX_RELATIVE_SENSITIVITY = 0.20

SELECTED_TRACKS = [2, 8, 63, 66, 125, 127]


def main():

    df = pd.read_csv(INPUT_PATH)

    required = {
        "frame",
        "track_id",
        "lane_status",
        "distance_dynamic_m",
        "relative_distance_sensitivity",
        "closing_speed_lr_5_m_s",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.duplicated(["track_id", "frame"]).any():
        raise ValueError("Duplicate frame/track observations")

    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    # A distance is low-sensitivity only when
    # all VP perturbation scenarios were valid
    # and their relative spread is <= 20%.
    df["point_low_sensitivity"] = (
        df["distance_dynamic_m"].notna()
        & df["relative_distance_sensitivity"].notna()
        & (
            df["relative_distance_sensitivity"]
            <= MAX_RELATIVE_SENSITIVITY
        )
    )

    df["window_complete"] = False
    df["window_low_sensitivity"] = False
    df["window_max_sensitivity"] = np.nan

    # Evaluate each track separately.
    for _, track in df.groupby(
        "track_id",
        sort=False,
    ):

        window_indices = []
        previous_frame = None

        for idx, row in track.iterrows():

            frame = int(row["frame"])

            if (
                previous_frame is None
                or frame != previous_frame + 1
            ):
                window_indices = []

            previous_frame = frame

            # Missing distances break the window.
            if pd.isna(row["distance_dynamic_m"]):
                window_indices = []
                continue

            window_indices.append(idx)

            if len(window_indices) > WINDOW:
                window_indices.pop(0)

            if len(window_indices) < WINDOW:
                continue

            window = df.loc[window_indices]

            df.at[idx, "window_complete"] = True

            sensitivities = window[
                "relative_distance_sensitivity"
            ]

            if sensitivities.notna().all():

                df.at[
                    idx,
                    "window_max_sensitivity"
                ] = sensitivities.max()

                df.at[
                    idx,
                    "window_low_sensitivity"
                ] = bool(
                    window["point_low_sensitivity"].all()
                )

    # Check agreement with the existing LR_5
    # estimator's availability.
    lr5_available = (
        df["closing_speed_lr_5_m_s"].notna()
    )

    assert (
        df["window_complete"].to_numpy()
        == lr5_available.to_numpy()
    ).all(), (
        "Window reconstruction does not match "
        "the existing LR_5 estimator"
    )

    # Diagnostic reliability categories.
    df["motion_quality"] = "UNAVAILABLE"

    df.loc[
        lr5_available,
        "motion_quality"
    ] = "UNCERTAIN"

    df.loc[
        lr5_available
        & df["window_low_sensitivity"],
        "motion_quality"
    ] = "LOW_SENSITIVITY"

    # This is only a scenario-based quality label,
    # not a calibrated confidence estimate.
    df["closing_speed_qualified_m_s"] = (
        df["closing_speed_lr_5_m_s"].where(
            df["motion_quality"] == "LOW_SENSITIVITY"
        )
    )

    # Independent diagnostic flag.
    df["large_apparent_speed"] = (
        lr5_available
        & (
            df["closing_speed_lr_5_m_s"].abs()
            > 30.0
        )
    )

    # ------------------------------------------
    # Results
    # ------------------------------------------

    print("\n--- MOTION RELIABILITY EVALUATION ---")

    print(f"Observations: {len(df)}")

    print("\nQuality distribution:")
    print(
        df["motion_quality"]
        .value_counts()
        .to_string()
    )

    print(
        "\nLarge apparent-speed estimates:",
        int(df["large_apparent_speed"].sum())
    )

    qualified = df[
        df["motion_quality"] == "LOW_SENSITIVITY"
    ]

    print(
        "\nQualified estimates:",
        len(qualified)
    )

    if not qualified.empty:

        print(
            "Median absolute closing speed:",
            f"{qualified['closing_speed_qualified_m_s'].abs().median():.2f} m/s"
        )

        print(
            "P95 absolute closing speed:",
            f"{qualified['closing_speed_qualified_m_s'].abs().quantile(.95):.2f} m/s"
        )

        print("\nLane classifications:")
        print(
            qualified["lane_status"]
            .value_counts()
            .to_string()
        )

    # ------------------------------------------
    # Selected tracks
    # ------------------------------------------

    print("\n--- SELECTED TRACKS ---")

    for track_id in SELECTED_TRACKS:

        track = df[
            df["track_id"] == track_id
        ]

        if track.empty:
            continue

        print(f"\nTrack {track_id}")

        print(
            track["motion_quality"]
            .value_counts()
            .to_string()
        )

    # ------------------------------------------
    # Save
    # ------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nReliability experiment complete!")


if __name__ == "__main__":
    main()
