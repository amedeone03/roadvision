
from pathlib import Path

import pandas as pd

from src.risk.lane_evidence import (
    classify_lane_evidence,
)


INPUT = Path("runs/phase6/ttc_proxy_results.csv")
OUTPUT_DIR = Path("runs/phase6")

OUTPUT_OBSERVATIONS = (
    OUTPUT_DIR / "lane_evidence_results.csv"
)
OUTPUT_EVENTS = (
    OUTPUT_DIR / "lane_approach_events.csv"
)

MIN_CONSECUTIVE_FRAMES = 3
FPS = 29.97


def main():
    df = pd.read_csv(INPUT)

    required = {
        "frame",
        "track_id",
        "lane_status",
        "approach_t015",
        "tau_proxy_s",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    keys = ["track_id", "frame"]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate frame/track observations"
        )

    df = df.sort_values(keys).reset_index(drop=True)

    df["evidence"] = [
        classify_lane_evidence(motion, lane)
        for motion, lane in zip(
            df["approach_t015"],
            df["lane_status"],
        )
    ]

    # Require a valid expansion proxy for
    # every EGO_LANE_APPROACH observation.
    ego_approach = (
        df["evidence"] == "EGO_LANE_APPROACH"
    )

    assert df.loc[
        ego_approach, "tau_proxy_s"
    ].notna().all()

    # Causal consecutive-frame streak counter.
    df["approach_streak"] = 0
    df["sustained_ego_approach"] = False

    event_rows = []

    for track_id, track in df.groupby(
        "track_id", sort=False
    ):
        streak = 0
        last_frame = None
        event_start = None
        event_taus = []

        def close_event():
            if event_start is None:
                return

            if streak >= MIN_CONSECUTIVE_FRAMES:
                event_rows.append({
                    "track_id": track_id,
                    "start_frame": event_start,
                    "end_frame": last_frame,
                    "duration_frames": streak,
                    "duration_s": streak / FPS,
                    "min_tau_proxy_s": min(
                        event_taus
                    ),
                    "median_tau_proxy_s": float(
                        pd.Series(event_taus).median()
                    ),
                })

        for idx, row in track.iterrows():
            frame = int(row["frame"])

            is_candidate = (
                row["evidence"]
                == "EGO_LANE_APPROACH"
            )

            continues = (
                is_candidate
                and last_frame is not None
                and frame == last_frame + 1
                and streak > 0
            )

            if not continues:
                close_event()

                streak = 0
                event_start = None
                event_taus = []

            if is_candidate:
                if streak == 0:
                    event_start = frame

                streak += 1
                event_taus.append(
                    float(row["tau_proxy_s"])
                )

                df.at[idx, "approach_streak"] = streak

                df.at[
                    idx,
                    "sustained_ego_approach"
                ] = (
                    streak >= MIN_CONSECUTIVE_FRAMES
                )

            last_frame = frame

        close_event()

    events = pd.DataFrame(
        event_rows,
        columns=[
            "track_id",
            "start_frame",
            "end_frame",
            "duration_frames",
            "duration_s",
            "min_tau_proxy_s",
            "median_tau_proxy_s",
        ],
    )

    # -------------------------------
    # SUMMARY
    # -------------------------------

    print("\n--- LANE-AWARE APPROACH EVIDENCE ---")

    print("Observations:", len(df))
    print("Unique tracks:", df["track_id"].nunique())

    print("\nEvidence categories:")
    print(
        df["evidence"]
        .value_counts()
        .to_string()
    )

    print("\n--- EGO-LANE APPROACH ---")

    ego = df[
        df["evidence"] == "EGO_LANE_APPROACH"
    ]

    print("Candidate observations:", len(ego))
    print(
        "Candidate tracks:",
        ego["track_id"].nunique()
    )

    sustained = df[
        df["sustained_ego_approach"]
    ]

    print(
        "Sustained observations:",
        len(sustained)
    )

    print(
        "Sustained tracks:",
        sustained["track_id"].nunique()
    )

    print(
        "Completed sustained episodes:",
        len(events)
    )

    if not events.empty:
        print("\nSustained episodes:")
        print(
            events.sort_values(
                "duration_frames",
                ascending=False,
            )
            .round(3)
            .to_string(index=False)
        )

    # -------------------------------
    # SAVE
    # -------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_OBSERVATIONS,
        index=False,
    )

    events.to_csv(
        OUTPUT_EVENTS,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_OBSERVATIONS)
    print(OUTPUT_EVENTS)

    print("\nLane evidence evaluation complete!")


if __name__ == "__main__":
    main()
