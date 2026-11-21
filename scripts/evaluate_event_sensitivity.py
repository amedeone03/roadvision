
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.event_stabilizer import (
    ApproachEventStabilizer,
)


# ==================================================
# CONFIGURATION
# ==================================================

INPUT = Path("runs/phase6/approach_results.csv")

BASELINE_EVENTS = Path(
    "runs/phase6/stabilized_approach_events.csv"
)

OUTPUT = Path("runs/phase7/sensitivity")

THRESHOLDS = [0.10, 0.15, 0.20]
CONFIRMATIONS = [2, 3, 4]
MAX_GAPS = [0, 1, 2]

BASELINE = (0.15, 3, 2)

FPS = 29.97

STATES = {
    "NONE",
    "PENDING",
    "CONFIRMED",
    "HELD",
}


# ==================================================
# STATE EXTRACTION
# ==================================================

def state_name(value):
    """
    Accept strings or enum-like state values.
    """
    if isinstance(value, str):
        name = value.split(".")[-1].upper()
        return name if name in STATES else None

    if hasattr(value, "value"):
        result = state_name(value.value)
        if result is not None:
            return result

    if hasattr(value, "name"):
        result = state_name(value.name)
        if result is not None:
            return result

    return None


def extract_state(result, stabilizer):
    """
    Normalize state returned by the existing
    Phase 6 stabilizer.

    Fail explicitly if its interface differs
    from the supported formats.
    """
    candidates = [result]

    if isinstance(result, dict):
        for key in ["event_state", "state"]:
            if key in result:
                candidates.append(result[key])

    if isinstance(result, (tuple, list)):
        candidates.extend(result)

    for obj in [result, stabilizer]:
        for name in [
            "event_state",
            "state",
            "current_state",
        ]:
            if hasattr(obj, name):
                candidates.append(getattr(obj, name))

    for candidate in candidates:
        state = state_name(candidate)
        if state is not None:
            return state

    raise RuntimeError(
        "Could not extract the stabilizer state.\n"
        f"update() returned: {result!r}\n"
        f"Available attributes: "
        f"{list(vars(stabilizer))}\n"
        "Inspect the existing stabilizer's return "
        "value before continuing."
    )


# ==================================================
# LOAD INPUT DATA
# ==================================================

def load_data():
    df = pd.read_csv(INPUT)

    required = {
        "frame",
        "track_id",
        "lane_status",
        "growth_robust_5",
        "approach_t010",
        "approach_t015",
        "approach_t020",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.duplicated(["frame", "track_id"]).any():
        raise ValueError(
            "Duplicate observations per track and frame"
        )

    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    return df


def approach_column(threshold):
    columns = {
        0.10: "approach_t010",
        0.15: "approach_t015",
        0.20: "approach_t020",
    }

    return columns[round(threshold, 2)]


# ==================================================
# RUN THE ORIGINAL STABILIZER
# ==================================================

def run_configuration(
    data,
    threshold,
    min_confirmations,
    max_gap,
):
    """
    Replay the existing Phase 6 state machine.

    Reconstruct completed episode boundaries
    from its online states, preserving the
    last positive frame as the episode end.
    """

    motion_column = approach_column(threshold)

    events = []
    state_counts = {
        state: 0 for state in STATES
    }

    for track_id, track in data.groupby(
        "track_id", sort=True
    ):
        stabilizer = ApproachEventStabilizer(
            min_confirmations=min_confirmations,
            max_gap=max_gap,
        )

        active_start = None
        confirmed_frame = None
        last_positive = None
        positive_frames = 0

        previous_frame = None

        def finish_event():
            if (
                active_start is not None
                and confirmed_frame is not None
                and last_positive is not None
            ):
                events.append({
                    "track_id": int(track_id),
                    "start_frame": int(active_start),
                    "confirmation_frame": int(
                        confirmed_frame
                    ),
                    "end_frame": int(last_positive),
                    "duration_frames": int(
                        last_positive - active_start + 1
                    ),
                    "positive_frames": int(
                        positive_frames
                    ),
                })

        for row in track.itertuples(index=False):
            frame = int(row.frame)

            # The original logic treats a missing
            # frame as a break in evidence.
            if (
                previous_frame is not None
                and frame > previous_frame + 1
            ):
                finish_event()

                active_start = None
                confirmed_frame = None
                last_positive = None
                positive_frames = 0

                stabilizer = ApproachEventStabilizer(
                    min_confirmations=min_confirmations,
                    max_gap=max_gap,
                )

            motion = getattr(row, motion_column)
            lane = row.lane_status

            growth = row.growth_robust_5

            tau = None
            if (
                pd.notna(growth)
                and float(growth) > threshold
            ):
                tau = 1.0 / float(growth)

            result = stabilizer.update(
                frame=frame,
                motion_state=motion,
                lane_status=lane,
                tau_proxy=tau,
            )

            state = extract_state(
                result,
                stabilizer,
            )

            state_counts[state] += 1

            positive = (
                motion == "APPROACHING"
                and lane == "IN_LANE"
            )

            if state == "NONE":
                finish_event()

                active_start = None
                confirmed_frame = None
                last_positive = None
                positive_frames = 0

            else:
                if active_start is None:
                    active_start = frame

                if positive:
                    last_positive = frame
                    positive_frames += 1

                if (
                    state == "CONFIRMED"
                    and confirmed_frame is None
                ):
                    confirmed_frame = frame

            previous_frame = frame

        # Flush episode at the end of each track.
        finish_event()

    columns = [
        "track_id",
        "start_frame",
        "confirmation_frame",
        "end_frame",
        "duration_frames",
        "positive_frames",
    ]

    events_df = pd.DataFrame(
        events,
        columns=columns,
    )

    if not events_df.empty:
        events_df = events_df.sort_values(
            ["start_frame", "track_id"]
        ).reset_index(drop=True)

    return events_df, state_counts


# ==================================================
# VERIFY BASELINE REPRODUCTION
# ==================================================

def verify_baseline(replayed):
    original = pd.read_csv(BASELINE_EVENTS)

    keys = [
        "track_id",
        "start_frame",
        "confirmation_frame",
        "end_frame",
    ]

    for key in keys:
        if key not in original.columns:
            raise ValueError(
                f"Missing baseline event column: {key}"
            )

    expected = sorted(
        tuple(map(int, row))
        for row in original[keys].itertuples(
            index=False, name=None
        )
    )

    actual = sorted(
        tuple(map(int, row))
        for row in replayed[keys].itertuples(
            index=False, name=None
        )
    )

    if actual != expected:
        print("\nBASELINE REPRODUCTION FAILED")

        print("\nExpected:")
        for event in expected:
            print(event)

        print("\nReplayed:")
        for event in actual:
            print(event)

        raise RuntimeError(
            "The sensitivity replay does not "
            "reproduce the original Phase 6 "
            "event boundaries. Do not interpret "
            "the sensitivity results until "
            "this discrepancy is resolved."
        )

    print(
        "\nBaseline reproduction PASSED:"
        f" {len(expected)} episodes match."
    )


# ==================================================
# EVENT COMPARISON
# ==================================================

def event_overlap_count(reference, candidate):
    """
    Count reference episodes that overlap at least
    one candidate episode from the same track.

    This is not event precision or recall.
    """

    count = 0

    for ref in reference.itertuples(index=False):
        found = False

        for event in candidate.itertuples(index=False):
            if event.track_id != ref.track_id:
                continue

            overlap = (
                event.start_frame <= ref.end_frame
                and event.end_frame >= ref.start_frame
            )

            if overlap:
                found = True
                break

        if found:
            count += 1

    return count


# ==================================================
# MAIN EXPERIMENT
# ==================================================

def main():
    data = load_data()

    print("\n--- PHASE 7.6 SENSITIVITY ---")
    print("Input observations:", len(data))
    print("Unique track IDs:", data.track_id.nunique())

    # First verify that our replay is equivalent
    # to the original Phase 6 experiment.
    baseline_events, baseline_states = (
        run_configuration(
            data,
            threshold=BASELINE[0],
            min_confirmations=BASELINE[1],
            max_gap=BASELINE[2],
        )
    )

    verify_baseline(baseline_events)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    all_events = []

    for threshold, confirmations, gap in product(
        THRESHOLDS,
        CONFIRMATIONS,
        MAX_GAPS,
    ):
        events, states = run_configuration(
            data,
            threshold=threshold,
            min_confirmations=confirmations,
            max_gap=gap,
        )

        is_baseline = (
            round(threshold, 2),
            confirmations,
            gap,
        ) == BASELINE

        if not events.empty:
            delays = (
                events["confirmation_frame"]
                - events["start_frame"]
            )

            mean_delay = float(delays.mean())
            longest_event = int(
                events["duration_frames"].max()
            )
            unique_tracks = int(
                events["track_id"].nunique()
            )
        else:
            mean_delay = np.nan
            longest_event = 0
            unique_tracks = 0

        overlap_count = event_overlap_count(
            baseline_events,
            events,
        )

        summary_rows.append({
            "growth_threshold": threshold,
            "min_confirmations": confirmations,
            "max_gap": gap,
            "baseline": is_baseline,
            "episodes": len(events),
            "unique_tracks": unique_tracks,
            "mean_delay_frames": mean_delay,
            "longest_event_frames": longest_event,
            "baseline_episodes_overlapped": (
                overlap_count
            ),
            "state_none": states["NONE"],
            "state_pending": states["PENDING"],
            "state_confirmed": states["CONFIRMED"],
            "state_held": states["HELD"],
        })

        for record in events.to_dict("records"):
            all_events.append({
                "growth_threshold": threshold,
                "min_confirmations": confirmations,
                "max_gap": gap,
                **record,
            })

        print(
            f"threshold={threshold:.2f} | "
            f"confirmations={confirmations} | "
            f"max_gap={gap} | "
            f"episodes={len(events):2d} | "
            f"tracks={unique_tracks:2d} | "
            f"baseline_overlap={overlap_count}/8"
        )

    # ------------------------------------------
    # Export
    # ------------------------------------------

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT / "sensitivity_summary.csv",
        index=False,
    )

    pd.DataFrame(all_events).to_csv(
        OUTPUT / "sensitivity_events.csv",
        index=False,
    )

    print("\n--- BASELINE ---")
    print(
        summary[summary["baseline"]]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- THRESHOLD SENSITIVITY ---")
    fixed = summary[
        (summary["min_confirmations"] == 3)
        & (summary["max_gap"] == 2)
    ]

    print(
        fixed[
            [
                "growth_threshold",
                "episodes",
                "unique_tracks",
                "mean_delay_frames",
                "baseline_episodes_overlapped",
            ]
        ].round(3).to_string(index=False)
    )

    print("\n--- CONFIRMATION SENSITIVITY ---")
    fixed = summary[
        (summary["growth_threshold"] == 0.15)
        & (summary["max_gap"] == 2)
    ]

    print(
        fixed[
            [
                "min_confirmations",
                "episodes",
                "unique_tracks",
                "mean_delay_frames",
            ]
        ].round(3).to_string(index=False)
    )

    print("\n--- GAP SENSITIVITY ---")
    fixed = summary[
        (summary["growth_threshold"] == 0.15)
        & (summary["min_confirmations"] == 3)
    ]

    print(
        fixed[
            [
                "max_gap",
                "episodes",
                "unique_tracks",
                "mean_delay_frames",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nSaved results to:", OUTPUT)


if __name__ == "__main__":
    main()
