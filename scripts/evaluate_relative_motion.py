
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "runs/motion/distance_stability_results.csv"
)

OUTPUT_PATH = Path(
    "runs/motion/relative_motion_results.csv"
)

FPS = 29.97
WINDOWS = [5, 9]
SELECTED_TRACKS = [2, 8, 63, 66, 125, 127]


def local_range_rate(window, fps):
    """
    Estimate dZ/dt by linear regression.

    The input contains consecutive valid
    observations from a single track.

    Negative slope = approaching.
    Positive slope = receding.
    """

    frames = window["frame"].to_numpy(dtype=float)

    distances = window[
        "distance_dynamic_m"
    ].to_numpy(dtype=float)

    if not np.isfinite(distances).all():
        return np.nan

    times = (frames - frames[0]) / fps

    # Center time for numerical stability.
    times = times - times.mean()

    denominator = np.sum(times ** 2)

    if denominator <= 0:
        return np.nan

    slope = np.sum(
        times * (distances - distances.mean())
    ) / denominator

    return float(slope)


def add_motion_estimates(df):

    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    for size in WINDOWS:
        df[f"closing_speed_lr_{size}_m_s"] = np.nan

    # Restrict each regression to uninterrupted
    # runs with valid distance estimates.
    for _, track in df.groupby("track_id", sort=False):

        valid_window = []
        previous_frame = None

        for idx, row in track.iterrows():

            frame = int(row["frame"])
            z = row["distance_dynamic_m"]

            if (
                previous_frame is None
                or frame != previous_frame + 1
                or pd.isna(z)
            ):
                valid_window = []

            previous_frame = frame

            if pd.isna(z):
                continue

            valid_window.append(idx)

            if len(valid_window) > max(WINDOWS):
                valid_window.pop(0)

            for size in WINDOWS:

                if len(valid_window) < size:
                    continue

                indices = valid_window[-size:]
                window = df.loc[indices]

                range_rate = local_range_rate(
                    window, FPS
                )

                df.at[
                    idx,
                    f"closing_speed_lr_{size}_m_s"
                ] = -range_rate

    return df


def summarize(values, label):

    values = values.dropna()

    print(f"\n{label}")
    print(f"Valid estimates: {len(values)}")

    if values.empty:
        return

    print(
        f"Median signed: {values.median():+.2f} m/s"
    )
    print(
        f"Median absolute: "
        f"{values.abs().median():.2f} m/s"
    )
    print(
        f"P95 absolute: "
        f"{values.abs().quantile(0.95):.2f} m/s"
    )


def main():

    df = pd.read_csv(INPUT_PATH)

    required = {
        "frame",
        "track_id",
        "distance_dynamic_m",
        "apparent_relative_velocity_m_s",
        "relative_distance_sensitivity",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    print("\n--- ROADVISION RELATIVE MOTION ---")
    print(f"Observations: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Video FPS: {FPS}")

    df = add_motion_estimates(df)

    # Raw closing speed is the negative of
    # the previous forward-distance derivative.
    df["closing_speed_raw_m_s"] = (
        -df["apparent_relative_velocity_m_s"]
    )

    # Diagnostic uncertainty metadata:
    # The current estimate may have a valid
    # central distance but incomplete sensitivity.
    df["distance_uncertainty_complete"] = (
        df["relative_distance_sensitivity"].notna()
    )

    df["distance_low_sensitivity"] = (
        df["distance_uncertainty_complete"]
        & (
            df["relative_distance_sensitivity"] <= 0.20
        )
    )

    # Compare all estimators on the same rows.
    columns = [
        "closing_speed_raw_m_s",
        "closing_speed_lr_5_m_s",
        "closing_speed_lr_9_m_s",
    ]

    matched = df.dropna(subset=columns)

    print("\n--- ESTIMATOR AVAILABILITY ---")

    for column in columns:
        print(
            f"{column}: "
            f"{df[column].notna().sum()}/{len(df)}"
        )

    print("\n--- GLOBAL MOTION ESTIMATES ---")

    for column in columns:
        summarize(df[column], column)

    print("\n--- MATCHED-OBSERVATION COMPARISON ---")
    print(f"Matched observations: {len(matched)}")

    for column in columns:
        summarize(matched[column], column)

    # Compare the frame-to-frame variation
    # in estimates on matched consecutive rows.
    print("\n--- MATCHED TEMPORAL STABILITY ---")

    matched = matched.copy()

    matched["previous_frame"] = (
        matched.groupby("track_id")["frame"].shift(1)
    )

    consecutive = (
        matched["frame"]
        - matched["previous_frame"] == 1
    )

    print(
        "Matched consecutive pairs:",
        int(consecutive.sum())
    )

    for column in columns:

        change = (
            matched.groupby("track_id")[column]
            .diff()
            .abs()
        )

        change = change[consecutive].dropna()

        if change.empty:
            continue

        print(
            f"{column}: "
            f"median_change={change.median():.2f}, "
            f"p95_change={change.quantile(.95):.2f} "
            f"m/s"
        )

    print("\n--- SELECTED TRACKS ---")

    for track_id in SELECTED_TRACKS:

        track = df[df["track_id"] == track_id]

        if track.empty:
            continue

        print(f"\nTrack {track_id}")
        print(f"Observations: {len(track)}")

        for column in columns:

            values = track[column].dropna()

            if values.empty:
                print(f"  {column}: unavailable")
                continue

            print(
                f"  {column}: "
                f"n={len(values)}, "
                f"median={values.median():+.2f}, "
                f"p95_abs={values.abs().quantile(.95):.2f}"
            )

    print("\n--- UNCERTAINTY COVERAGE ---")

    print(
        "Central distances:",
        int(df["distance_dynamic_m"].notna().sum())
    )

    print(
        "Complete sensitivity scenarios:",
        int(df["distance_uncertainty_complete"].sum())
    )

    print(
        "Low-sensitivity observations:",
        int(df["distance_low_sensitivity"].sum())
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nRelative motion experiment complete!")


if __name__ == "__main__":
    main()
