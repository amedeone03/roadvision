"""Replay the integrated tracking CSV through the per-track processor.

The processor does not know these paths. This test supplies the
stored boxes and lane labels, then compares the causal outputs
with the Phase 6 reference tables.
"""

import math
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
import pytest

from roadvision.config import (
    PHASE6_DURATION_DIVISOR,
    REPLAY_ABSOLUTE_TOLERANCE,
    load_config,
)
from roadvision.tracks import TrackMotionProcessor


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "videos" / "test_drive.mp4"
OBSERVATIONS = ROOT / "runs" / "integrated_final.csv"
REFERENCE = ROOT / "runs" / "phase6" / "stabilized_evidence_results.csv"
EVENTS = ROOT / "runs" / "phase6" / "stabilized_approach_events.csv"

EXPECTED_ROWS = 2656
EXPECTED_STATE_COUNTS = {
    "NONE": 2559,
    "PENDING": 62,
    "CONFIRMED": 25,
    "HELD": 10,
}
GROWTH_TOLERANCE = REPLAY_ABSOLUTE_TOLERANCE["growth_robust_5"]
TAU_TOLERANCE = REPLAY_ABSOLUTE_TOLERANCE["tau_proxy_s"]


def _require_reference_files():
    missing = [
        path
        for path in (VIDEO, OBSERVATIONS, REFERENCE, EVENTS)
        if not path.is_file()
    ]

    if missing:
        pytest.skip("Phase 6 replay files are not in this checkout")


def _measured_fps() -> float:
    capture = cv2.VideoCapture(str(VIDEO))

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0.0:
        raise AssertionError(f"Invalid measured FPS: {fps}")

    return fps


def _missing(value) -> bool:
    if value is None:
        return True

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isnan(number)


def _assert_float_pair(reference, actual, tolerance: float, label: str):
    reference_missing = _missing(reference)
    actual_missing = _missing(actual)

    if reference_missing or actual_missing:
        assert reference_missing and actual_missing, (
            f"{label}: missing values disagree "
            f"(reference={reference!r}, actual={actual!r})"
        )
        return

    difference = abs(float(reference) - float(actual))
    assert difference <= tolerance, (
        f"{label}: |{float(reference)} - {float(actual)}| = {difference}"
    )


def test_phase6_replay_matches_stabilized_outputs():
    _require_reference_files()

    fps = _measured_fps()
    source = pd.read_csv(OBSERVATIONS)
    reference = pd.read_csv(REFERENCE)
    reference_events = pd.read_csv(EVENTS)

    keys = ["frame", "track_id"]

    assert len(source) == EXPECTED_ROWS
    assert len(reference) == EXPECTED_ROWS
    assert not source.duplicated(keys).any()
    assert not reference.duplicated(keys).any()

    lane_check = source[keys + ["lane_status"]].merge(
        reference[keys + ["lane_status"]],
        on=keys,
        how="outer",
        suffixes=("_source", "_reference"),
        indicator=True,
    )
    assert lane_check["_merge"].eq("both").all()
    assert lane_check["lane_status_source"].eq(
        lane_check["lane_status_reference"]
    ).all()

    processor = TrackMotionProcessor(fps, load_config())
    estimates = []

    ordered = source.sort_values(keys)
    for row in ordered.itertuples(index=False):
        estimates.append(
            processor.update(
                frame=int(row.frame),
                track_id=int(row.track_id),
                bbox=(row.x1, row.y1, row.x2, row.y2),
                lane_status=str(row.lane_status),
            )
        )

    episodes = processor.finish()
    actual = pd.DataFrame([asdict(estimate) for estimate in estimates])

    compared = reference[
        keys
        + [
            "approach_t015",
            "evidence",
            "event_state",
            "growth_robust_5",
            "tau_proxy_s",
        ]
    ].merge(
        actual[
            keys
            + [
                "approach_state",
                "evidence",
                "event_state",
                "growth_robust_5",
                "tau_proxy_s",
            ]
        ],
        on=keys,
        how="outer",
        suffixes=("_reference", "_actual"),
        indicator=True,
    )

    assert compared["_merge"].eq("both").all()
    assert len(compared) == EXPECTED_ROWS

    categorical = {
        "approach_t015": "approach_state",
        "evidence_reference": "evidence_actual",
        "event_state_reference": "event_state_actual",
    }

    for reference_column, actual_column in categorical.items():
        mismatch = compared[reference_column] != compared[actual_column]
        assert not mismatch.any(), (
            f"{reference_column} disagrees on "
            f"{int(mismatch.sum())} rows. First rows:\n"
            f"{compared.loc[mismatch, keys + [reference_column, actual_column]].head(8)}"
        )

    counts = actual["event_state"].value_counts().to_dict()
    assert counts.get("NONE", 0) == EXPECTED_STATE_COUNTS["NONE"]
    assert counts.get("PENDING", 0) == EXPECTED_STATE_COUNTS["PENDING"]
    assert counts.get("CONFIRMED", 0) == EXPECTED_STATE_COUNTS["CONFIRMED"]
    assert counts.get("HELD", 0) == EXPECTED_STATE_COUNTS["HELD"]

    for index, row in compared.iterrows():
        identity = f"frame {int(row.frame)} track {int(row.track_id)}"
        _assert_float_pair(
            row["growth_robust_5_reference"],
            row["growth_robust_5_actual"],
            GROWTH_TOLERANCE,
            f"growth_robust_5 {identity}",
        )
        _assert_float_pair(
            row["tau_proxy_s_reference"],
            row["tau_proxy_s_actual"],
            TAU_TOLERANCE,
            f"tau_proxy_s {identity}",
        )

    assert len(episodes) == 8
    assert len(reference_events) == 8

    actual_events = pd.DataFrame(
        [asdict(episode) for episode in episodes]
    ).sort_values(
        ["start_frame", "track_id"]
    ).reset_index(drop=True)
    reference_events = reference_events.sort_values(
        ["start_frame", "track_id"]
    ).reset_index(drop=True)

    exact_event_columns = [
        "track_id",
        "start_frame",
        "end_frame",
        "confirmation_frame",
        "duration_frames",
        "positive_frames",
        "interruption_frames",
    ]

    for column in exact_event_columns:
        assert actual_events[column].astype(int).tolist() == (
            reference_events[column].astype(int).tolist()
        ), column

    for index, row in reference_events.iterrows():
        _assert_float_pair(
            row["min_tau_proxy_s"],
            actual_events.loc[index, "min_tau_proxy_s"],
            TAU_TOLERANCE,
            f"min_tau_proxy_s track {int(row.track_id)} "
            f"start {int(row.start_frame)}",
        )

    # The reference duration divides by 29.97. The processor
    # divides by the measured frame rate. Frame counts above
    # already match, so this only records the two conventions.
    assert actual_events["duration_s"].tolist() == pytest.approx(
        (actual_events["duration_frames"] / fps).tolist()
    )
    assert reference_events["duration_s"].tolist() == pytest.approx(
        (reference_events["duration_frames"] / PHASE6_DURATION_DIVISOR).tolist()
    )
