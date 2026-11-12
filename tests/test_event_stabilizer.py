
from src.risk.event_stabilizer import (
    ApproachEventStabilizer,
)


def test_three_confirmations():
    f = ApproachEventStabilizer()

    states = [
        f.update(i, "APPROACHING", "IN_LANE", 3.0)
        for i in [1, 2, 3]
    ]

    assert states == [
        "PENDING", "PENDING", "CONFIRMED"
    ]


def test_short_unknown_gap():
    f = ApproachEventStabilizer()

    results = [
        f.update(1, "APPROACHING", "IN_LANE", 3.0),
        f.update(2, "APPROACHING", "IN_LANE", 2.8),
        f.update(3, "APPROACHING", "UNKNOWN", 2.7),
        f.update(4, "APPROACHING", "IN_LANE", 2.6),
    ]

    assert results == [
        "PENDING",
        "PENDING",
        "PENDING",
        "CONFIRMED",
    ]


def test_short_stable_gap():
    f = ApproachEventStabilizer()

    for i in [1, 2, 3]:
        f.update(i, "APPROACHING", "IN_LANE", 3.0)

    assert f.update(
        4, "STABLE", "IN_LANE"
    ) == "HELD"

    assert f.update(
        5, "APPROACHING", "IN_LANE", 2.5
    ) == "CONFIRMED"

    events = f.finish()

    assert len(events) == 1
    assert events[0]["start_frame"] == 1
    assert events[0]["end_frame"] == 5
    assert events[0]["positive_frames"] == 4


def test_recession_ends_event():
    f = ApproachEventStabilizer()

    for i in [1, 2, 3]:
        f.update(i, "APPROACHING", "IN_LANE", 2.0)

    assert f.update(
        4, "RECEDING", "IN_LANE"
    ) == "NONE"

    events = f.finish()

    assert len(events) == 1
    assert events[0]["end_frame"] == 3


def test_out_lane_ends_event():
    f = ApproachEventStabilizer()

    for i in [1, 2, 3]:
        f.update(i, "APPROACHING", "IN_LANE", 3.0)

    assert f.update(
        4, "APPROACHING", "OUT_LANE", 2.0
    ) == "NONE"

    assert len(f.finish()) == 1


def test_long_gap_splits_events():
    f = ApproachEventStabilizer()

    for i in [1, 2, 3]:
        f.update(i, "APPROACHING", "IN_LANE", 3.0)

    for i in [4, 5, 6]:
        f.update(i, "STABLE", "IN_LANE")

    assert len(f.completed_events) == 1
    assert f.completed_events[0]["end_frame"] == 3


def test_missing_track_frame_breaks_continuity():
    f = ApproachEventStabilizer()

    for i in [1, 2, 3]:
        f.update(i, "APPROACHING", "IN_LANE", 3.0)

    assert f.update(
        6, "APPROACHING", "IN_LANE", 2.0
    ) == "PENDING"

    events = f.finish()

    assert len(events) == 1
    assert events[0]["end_frame"] == 3


def test_no_confirmation_no_event():
    f = ApproachEventStabilizer()

    f.update(1, "APPROACHING", "IN_LANE", 2.0)
    f.update(2, "RECEDING", "IN_LANE")

    assert f.finish() == []


def test_trailing_gap_not_counted_in_event():
    f = ApproachEventStabilizer(
        min_confirmations=3,
        max_gap=2,
    )

    for frame in [1, 2, 3]:
        f.update(
            frame,
            "APPROACHING",
            "IN_LANE",
            tau_proxy=3.0,
        )

    # Three nonpositive observations.
    # The third ends the event.
    for frame in [4, 5, 6]:
        f.update(
            frame,
            "STABLE",
            "IN_LANE",
        )

    events = f.finish()

    assert len(events) == 1
    assert events[0]["start_frame"] == 1
    assert events[0]["end_frame"] == 3
    assert events[0]["positive_frames"] == 3
    assert events[0]["interruption_frames"] == 0
