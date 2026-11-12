
import pytest

from src.risk.lane_evidence import (
    classify_lane_evidence,
)


@pytest.mark.parametrize(
    "lane,expected",
    [
        ("IN_LANE", "EGO_LANE_APPROACH"),
        ("OUT_LANE", "OTHER_LANE_APPROACH"),
        ("UNKNOWN", "LANE_UNCERTAIN_APPROACH"),
    ],
)
def test_approaching_by_lane(lane, expected):
    assert classify_lane_evidence(
        "APPROACHING", lane
    ) == expected


@pytest.mark.parametrize(
    "motion",
    ["STABLE", "RECEDING"],
)
def test_nonapproaching(motion):
    assert classify_lane_evidence(
        motion, "IN_LANE"
    ) == "NO_APPROACH_EVIDENCE"


def test_missing_motion():
    assert classify_lane_evidence(
        "UNKNOWN", "IN_LANE"
    ) == "MOTION_UNAVAILABLE"


def test_invalid_inputs():
    with pytest.raises(ValueError):
        classify_lane_evidence(
            "INVALID", "IN_LANE"
        )

    with pytest.raises(ValueError):
        classify_lane_evidence(
            "APPROACHING", "INVALID"
        )
