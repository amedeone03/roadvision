
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("runs/motion/dynamic_distance_results.csv")
OUTPUT = Path("runs/motion/distance_stability_results.csv")

FPS = 29.97
H = 1.5
FY = 1000.0
MIN_SEPARATION = 30.0

SELECTED_TRACKS = [2, 8, 63, 66, 125, 127]


def distance(y, horizon):
    if not (np.isfinite(y) and np.isfinite(horizon)):
        return np.nan

    separation = y - horizon

    if separation <= MIN_SEPARATION:
        return np.nan

    return H * FY / separation


def main():
    df = pd.read_csv(INPUT)

    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    grouped = df.groupby("track_id")

    for column in [
        "frame", "contact_y", "filtered_vp_y",
        "distance_dynamic_m"
    ]:
        df[f"previous_{column}"] = (
            grouped[column].shift(1)
        )

    # Only evaluate consecutive observations.
    consecutive = (
        df["frame"] - df["previous_frame"] == 1
    )

    valid_pair = (
        consecutive
        & df["distance_dynamic_m"].notna()
        & df["previous_distance_dynamic_m"].notna()
    )

    df["delta_distance_m"] = (
        df["distance_dynamic_m"]
        - df["previous_distance_dynamic_m"]
    ).where(valid_pair)

    # Exact decomposition:
    # 1. Move the contact point, holding previous VP.
    # 2. Move the VP, holding current contact point.
    #
    # Both terms must exist for decomposition.
    intermediate = np.array([
        distance(y, vp)
        for y, vp in zip(
            df["contact_y"],
            df["previous_filtered_vp_y"],
        )
    ])

    df["contact_contribution_m"] = (
        intermediate
        - df["previous_distance_dynamic_m"]
    )

    df["vp_contribution_m"] = (
        df["distance_dynamic_m"]
        - intermediate
    )

    df.loc[
        ~valid_pair,
        ["contact_contribution_m", "vp_contribution_m"]
    ] = np.nan

    decomposable = (
        valid_pair
        & df["contact_contribution_m"].notna()
        & df["vp_contribution_m"].notna()
    )

    df["vp_dominant"] = (
        decomposable
        & (
            df["vp_contribution_m"].abs()
            > df["contact_contribution_m"].abs()
        )
    )

    df["apparent_relative_velocity_m_s"] = (
        df["delta_distance_m"] * FPS
    )

    # A diagnostic subset with complete VP
    # sensitivity scenarios at both time steps.
    scenario_valid = (
        df["relative_distance_sensitivity"].notna()
        & grouped[
            "relative_distance_sensitivity"
        ].shift(1).notna()
    )

    df["scenario_complete_pair"] = (
        valid_pair & scenario_valid
    )

    # Check the exact decomposition numerically.
    residual = (
        df.loc[decomposable, "delta_distance_m"]
        - df.loc[decomposable, "contact_contribution_m"]
        - df.loc[decomposable, "vp_contribution_m"]
    )

    assert np.allclose(residual, 0.0, atol=1e-8)

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT, index=False)

    print("\n--- DISTANCE TRAJECTORY STABILITY ---")
    print("Observations:", len(df))
    print("Consecutive valid pairs:", int(valid_pair.sum()))
    print("Decomposable pairs:", int(decomposable.sum()))
    print("Complete-scenario pairs:",
          int(df["scenario_complete_pair"].sum()))
    print("VP-dominant pairs:",
          int(df["vp_dominant"].sum()))

    print("\n--- GLOBAL DISTANCE CHANGES ---")

    changes = df.loc[
        valid_pair, "delta_distance_m"
    ].abs()

    if not changes.empty:
        print(f"Median absolute change: {changes.median():.3f} m")
        print(f"P95 absolute change: {changes.quantile(.95):.3f} m")

    print("\n--- SELECTED TRACKS ---")

    for track_id in SELECTED_TRACKS:
        track = df[df["track_id"] == track_id]

        if track.empty:
            continue

        valid = track["delta_distance_m"].dropna()

        print(f"\nTrack {track_id}")
        print("Consecutive valid pairs:", len(valid))
        print("VP-dominant pairs:",
              int(track["vp_dominant"].sum()))

        if not valid.empty:
            print(
                "Median absolute change:",
                f"{valid.abs().median():.3f} m"
            )
            print(
                "P95 absolute change:",
                f"{valid.abs().quantile(.95):.3f} m"
            )

    print(f"\nSaved: {OUTPUT}")
    print("Distance stability analysis complete!")


if __name__ == "__main__":
    main()
