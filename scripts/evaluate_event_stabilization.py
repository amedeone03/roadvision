
from pathlib import Path

import pandas as pd

from src.risk.event_stabilizer import ApproachEventStabilizer


INPUT = Path("runs/phase6/lane_evidence_results.csv")
OLD_EVENTS = Path("runs/phase6/lane_approach_events.csv")

OUTPUT_DIR = Path("runs/phase6")
OUTPUT_OBSERVATIONS = OUTPUT_DIR / "stabilized_evidence_results.csv"
OUTPUT_EVENTS = OUTPUT_DIR / "stabilized_approach_events.csv"

FPS = 29.97
MIN_CONFIRMATIONS = 3
MAX_GAP = 2


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

    if df.duplicated(["track_id", "frame"]).any():
        raise ValueError(
            "Duplicate frame/track observations"
        )

    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    df["event_state"] = "NONE"

    all_events = []

    # Each track gets a separate state machine.
    for track_id, track in df.groupby(
        "track_id", sort=False
    ):
        stabilizer = ApproachEventStabilizer(
            min_confirmations=MIN_CONFIRMATIONS,
            max_gap=MAX_GAP,
        )

        for idx, row in track.iterrows():
            tau = (
                None
                if pd.isna(row["tau_proxy_s"])
                else float(row["tau_proxy_s"])
            )

            state = stabilizer.update(
                frame=int(row["frame"]),
                motion_state=row["approach_t015"],
                lane_status=row["lane_status"],
                tau_proxy=tau,
            )

            df.at[idx, "event_state"] = state

        completed = stabilizer.finish()

        for event in completed:
            event["track_id"] = int(track_id)
            event["duration_s"] = (
                event["duration_frames"] / FPS
            )
            all_events.append(event)

    event_columns = [
        "track_id",
        "start_frame",
        "end_frame",
        "confirmation_frame",
        "duration_frames",
        "duration_s",
        "positive_frames",
        "interruption_frames",
        "min_tau_proxy_s",
    ]

    events = pd.DataFrame(
        all_events,
        columns=event_columns,
    )

    if not events.empty:
        events = events.sort_values(
            ["start_frame", "track_id"]
        ).reset_index(drop=True)

    # =====================================
    # VALIDATE STATES
    # =====================================

    valid_states = {
        "NONE", "PENDING", "CONFIRMED", "HELD"
    }

    assert df["event_state"].isin(valid_states).all()

    confirmed = df[df["event_state"] == "CONFIRMED"]
    held = df[df["event_state"] == "HELD"]

    assert (
        confirmed["approach_t015"] == "APPROACHING"
    ).all()

    assert (
        confirmed["lane_status"] == "IN_LANE"
    ).all()

    assert (
        held["lane_status"] != "OUT_LANE"
    ).all()

    assert (
        held["approach_t015"] != "RECEDING"
    ).all()

    # =====================================
    # REPORT STATE DISTRIBUTION
    # =====================================

    print("\n--- EVENT STABILIZATION ---")
    print("Observations:", len(df))
    print("Unique tracks:", df["track_id"].nunique())

    print("\nState distribution:")
    print(
        df["event_state"]
        .value_counts()
        .reindex(
            ["NONE", "PENDING", "CONFIRMED", "HELD"],
            fill_value=0,
        )
        .to_string()
    )

    print("\n--- STABILIZED EPISODES ---")

    print("Completed confirmed episodes:", len(events))

    if not events.empty:
        print(
            "Tracks with confirmed episodes:",
            events["track_id"].nunique(),
        )

        print("\nEpisodes:")
        print(
            events[
                [
                    "track_id",
                    "start_frame",
                    "end_frame",
                    "confirmation_frame",
                    "duration_frames",
                    "positive_frames",
                    "interruption_frames",
                    "min_tau_proxy_s",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

    # =====================================
    # COMPARE WITH PHASE 6.3
    # =====================================

    if OLD_EVENTS.exists():
        old = pd.read_csv(OLD_EVENTS)

        print("\n--- PREVIOUS VS STABILIZED ---")
        print("Previous episodes:", len(old))
        print("Stabilized episodes:", len(events))

        if not old.empty:
            for track_id in sorted(
                set(old["track_id"])
                | set(events["track_id"])
            ):
                previous_count = int(
                    (old["track_id"] == track_id).sum()
                )
                new_count = int(
                    (events["track_id"] == track_id).sum()
                )

                print(
                    f"Track {track_id}: "
                    f"{previous_count} -> {new_count}"
                )

    # =====================================
    # SAVE
    # =====================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    print("\nEvent stabilization complete!")


if __name__ == "__main__":
    main()
