
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.time_to_contact import (
    estimate_tau_proxy,
)


INPUT = Path("runs/phase6/approach_results.csv")
OUTPUT = Path("runs/phase6/ttc_proxy_results.csv")

THRESHOLD = 0.15
SELECTED_TRACKS = [2, 8, 63, 66, 125, 127]


def main():

    df = pd.read_csv(INPUT)

    required = {
        "frame",
        "track_id",
        "growth_robust_5",
        "approach_t015",
        "lane_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.duplicated(["frame", "track_id"]).any():
        raise ValueError(
            "Duplicate frame/track observations"
        )

    df["tau_proxy_s"] = df[
        "growth_robust_5"
    ].apply(
        lambda value: estimate_tau_proxy(
            value,
            min_growth=THRESHOLD,
        )
    )

    # Ensure the proxy is available only
    # for APPROACHING observations.
    available = df["tau_proxy_s"].notna()

    assert (
        df.loc[
            available,
            "approach_t015",
        ] == "APPROACHING"
    ).all()

    print("\n--- EXPERIMENTAL TTC PROXY ---")

    print("Total observations:", len(df))
    print(
        "Available proxy estimates:",
        int(available.sum()),
    )

    print("\n--- GLOBAL PROXY STATISTICS ---")

    values = df["tau_proxy_s"].dropna()

    if not values.empty:
        print(
            values.describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).round(3).to_string()
        )

        for seconds in [1.0, 2.0, 3.0]:
            print(
                f"Proxy below {seconds:.0f} s:",
                int((values < seconds).sum()),
            )

    print("\n--- PROXY BY LANE ---")

    summary = df.groupby("lane_status").agg(
        observations=("track_id", "size"),
        available=("tau_proxy_s", "count"),
        median_tau_s=("tau_proxy_s", "median"),
    )

    print(summary.round(3).to_string())

    print("\n--- SELECTED TRACKS ---")

    for track_id in SELECTED_TRACKS:

        track = df[
            df["track_id"] == track_id
        ]

        if track.empty:
            continue

        tau = track["tau_proxy_s"].dropna()

        print(f"\nTrack {track_id}")
        print("Observations:", len(track))
        print("Available proxies:", len(tau))

        if not tau.empty:
            print(
                f"Median proxy: {tau.median():.2f} s"
            )
            print(
                f"Minimum proxy: {tau.min():.2f} s"
            )

    # Track-level instability diagnostic:
    # compare consecutive available estimates.
    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    previous_frame = (
        df.groupby("track_id")["frame"].shift(1)
    )

    previous_tau = (
        df.groupby("track_id")[
            "tau_proxy_s"
        ].shift(1)
    )

    consecutive = (
        df["frame"] - previous_frame == 1
    )

    df["abs_tau_change_s"] = (
        df["tau_proxy_s"] - previous_tau
    ).abs().where(consecutive)

    changes = df["abs_tau_change_s"].dropna()

    print("\n--- TEMPORAL CONSISTENCY ---")

    print("Consecutive proxy pairs:", len(changes))

    if not changes.empty:
        print(
            "Median absolute change:",
            f"{changes.median():.3f} s",
        )
        print(
            "P95 absolute change:",
            f"{changes.quantile(0.95):.3f} s",
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print("\nTTC proxy evaluation complete!")


if __name__ == "__main__":
    main()
