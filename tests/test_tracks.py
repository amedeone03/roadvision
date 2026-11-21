"""Unit tests for causal per-track motion state."""

import math

import pytest

from roadvision.tracks import TrackMotionProcessor
from src.motion.robust_growth import rolling_robust_growth


FPS = 10.0


def _box(height: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, 10.0, height)


def _expanding_height(frame: int, fps: float = FPS, rate: float = 1.0) -> float:
    """Box height whose log grows at ``rate`` per second."""

    return math.exp(rate * (frame - 1) / fps)


def test_empty_input_has_no_episodes():
    processor = TrackMotionProcessor(FPS)

    assert processor.finish() == ()


def test_short_track_has_no_growth_estimate():
    processor = TrackMotionProcessor(FPS)

    for frame in range(1, 5):
        sample = processor.update(
            frame,
            track_id=4,
            bbox=_box(40.0 + frame),
            lane_status="IN_LANE",
        )

        assert sample.growth_robust_5 is None
        assert sample.tau_proxy_s is None
        assert sample.approach_state == "UNKNOWN"
        assert sample.evidence == "MOTION_UNAVAILABLE"
        assert sample.event_state == "NONE"
        assert sample.time_seconds == pytest.approx((frame - 1) / FPS)

    assert processor.finish() == ()


def test_growth_matches_rolling_robust_growth():
    fps = 29.97002997002997
    frames = list(range(1, 12))
    heights = [80.0 + 2.5 * frame for frame in frames]
    processor = TrackMotionProcessor(fps)
    estimates = [
        processor.update(frame, 1, _box(height), "OUT_LANE")
        for frame, height in zip(frames, heights)
    ]
    expected = rolling_robust_growth(
        [(frame - 1) / fps for frame in frames],
        heights,
        window=5,
    )

    assert len(estimates) == len(expected)

    for sample, slope in zip(estimates, expected):
        if math.isnan(float(slope)):
            assert sample.growth_robust_5 is None
        else:
            assert sample.growth_robust_5 == slope


def test_missing_frame_resets_the_growth_window():
    processor = TrackMotionProcessor(FPS)
    segment_frames = [7, 8, 9, 10, 11]
    segment_heights = [40.0 + 3.0 * frame for frame in segment_frames]

    for frame in range(1, 6):
        sample = processor.update(
            frame,
            3,
            _box(_expanding_height(frame)),
            "OUT_LANE",
        )

    assert sample.growth_robust_5 is not None

    restarted = []

    for frame, height in zip(segment_frames, segment_heights):
        restarted.append(
            processor.update(frame, 3, _box(height), "OUT_LANE")
        )

    assert restarted[0].growth_robust_5 is None
    assert restarted[0].approach_state == "UNKNOWN"

    expected = rolling_robust_growth(
        [(frame - 1) / FPS for frame in segment_frames],
        segment_heights,
        window=5,
    )

    for sample, slope in zip(restarted, expected):
        if math.isnan(float(slope)):
            assert sample.growth_robust_5 is None
        else:
            assert sample.growth_robust_5 == slope


def test_a_frame_gap_closes_a_confirmed_episode():
    continuous = TrackMotionProcessor(FPS)
    broken = TrackMotionProcessor(FPS)

    for frame in range(1, 8):
        height = _box(_expanding_height(frame))
        continuous.update(frame, 3, height, "IN_LANE")
        broken.update(frame, 3, height, "IN_LANE")

    broken.update(9, 3, _box(_expanding_height(9)), "IN_LANE")

    gap_episodes = broken.finish()
    end_episodes = continuous.finish()

    assert len(gap_episodes) == 1
    assert len(end_episodes) == 1
    assert gap_episodes[0].start_frame == end_episodes[0].start_frame == 5
    assert gap_episodes[0].end_frame == end_episodes[0].end_frame == 7
    assert (
        gap_episodes[0].confirmation_frame
        == end_episodes[0].confirmation_frame
        == 7
    )


def test_track_histories_are_independent():
    processor = TrackMotionProcessor(FPS)
    flat_only = TrackMotionProcessor(FPS)

    for frame in range(1, 8):
        growing = processor.update(
            frame,
            track_id=1,
            bbox=_box(_expanding_height(frame)),
            lane_status="IN_LANE",
        )
        flat = processor.update(
            frame,
            track_id=2,
            bbox=_box(50.0),
            lane_status="IN_LANE",
        )
        control = flat_only.update(
            frame,
            track_id=2,
            bbox=_box(50.0),
            lane_status="IN_LANE",
        )

        if flat.growth_robust_5 is None:
            assert control.growth_robust_5 is None
        else:
            assert flat.growth_robust_5 == control.growth_robust_5

        assert flat.growth_robust_5 is None or flat.approach_state == "STABLE"
        assert flat.event_state == "NONE"
        assert growing.track_id == 1

    episodes = processor.finish()

    assert [episode.track_id for episode in episodes] == [1]
    assert flat_only.finish() == ()


def test_duplicate_and_reversed_frames_do_not_change_state():
    processor = TrackMotionProcessor(FPS)
    frames = list(range(1, 7))

    for frame in frames:
        processor.update(
            frame,
            5,
            _box(_expanding_height(frame)),
            "OUT_LANE",
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        processor.update(6, 5, _box(_expanding_height(6)), "OUT_LANE")

    with pytest.raises(ValueError, match="strictly increasing"):
        processor.update(4, 5, _box(_expanding_height(4)), "OUT_LANE")

    continued = processor.update(
        7,
        5,
        _box(_expanding_height(7)),
        "OUT_LANE",
    )
    control = TrackMotionProcessor(FPS)
    sample = None

    for frame in frames + [7]:
        sample = control.update(
            frame,
            5,
            _box(_expanding_height(frame)),
            "OUT_LANE",
        )

    assert sample is not None
    assert continued.growth_robust_5 == sample.growth_robust_5
    assert continued.approach_state == sample.approach_state


def test_video_end_finalizes_a_confirmed_episode():
    processor = TrackMotionProcessor(FPS)
    states = []

    for frame in range(1, 8):
        sample = processor.update(
            frame,
            6,
            _box(_expanding_height(frame)),
            lane_status="IN_LANE",
        )
        states.append(sample.event_state)

    assert states == [
        "NONE",
        "NONE",
        "NONE",
        "NONE",
        "PENDING",
        "PENDING",
        "CONFIRMED",
    ]

    episodes = processor.finish()

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.track_id == 6
    assert episode.start_frame == 5
    assert episode.end_frame == 7
    assert episode.confirmation_frame == 7
    assert episode.duration_frames == 3
    assert episode.positive_frames == 3
    assert episode.interruption_frames == 0
    assert episode.duration_s == pytest.approx(3 / FPS)
    assert episode.min_tau_proxy_s == pytest.approx(1.0)
    assert processor.finish() == episodes

    with pytest.raises(RuntimeError, match="finished"):
        processor.update(8, 6, _box(10.0), "IN_LANE")


def test_unknown_lane_is_not_confirmed_evidence():
    processor = TrackMotionProcessor(FPS)
    sample = None

    for frame in range(1, 8):
        sample = processor.update(
            frame,
            9,
            _box(_expanding_height(frame)),
            lane_status="UNKNOWN",
        )

    assert sample is not None
    assert sample.approach_state == "APPROACHING"
    assert sample.evidence == "LANE_UNCERTAIN_APPROACH"
    assert sample.event_state == "NONE"
    assert sample.tau_proxy_s == pytest.approx(1.0)
    assert processor.finish() == ()


def test_held_retains_context_without_new_approach_evidence():
    processor = TrackMotionProcessor(FPS)

    for frame in range(1, 8):
        processor.update(
            frame,
            11,
            _box(_expanding_height(frame)),
            "IN_LANE",
        )

    plateau = _expanding_height(7)
    observed = []

    for frame in range(8, 14):
        sample = processor.update(frame, 11, _box(plateau), "IN_LANE")
        observed.append((frame, sample.approach_state, sample.event_state))

    assert ("STABLE", "HELD") in [
        (approach, event) for _, approach, event in observed
    ]
    assert observed[-1][1:] == ("STABLE", "NONE")

    episodes = processor.finish()

    assert len(episodes) == 1
    assert episodes[0].confirmation_frame == 7
    assert episodes[0].start_frame == 5
    assert episodes[0].positive_frames >= 3


def test_processor_rejects_bad_fps_and_does_not_name_csv_files():
    with pytest.raises(ValueError, match="fps"):
        TrackMotionProcessor(0)

    source = open(
        TrackMotionProcessor.__init__.__code__.co_filename,
        encoding="utf-8",
    ).read()

    assert "integrated_final" not in source
    assert "stabilized_evidence" not in source
    assert "phase6" not in source.lower()
