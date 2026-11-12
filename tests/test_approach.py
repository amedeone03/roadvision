
import math

import pytest

from src.risk.approach import (
    classify_approach,
    APPROACHING,
    RECEDING,
    STABLE,
    UNKNOWN,
)


def test_approaching():
    assert classify_approach(0.45) == APPROACHING


def test_receding():
    assert classify_approach(-0.32) == RECEDING


def test_stable():
    assert classify_approach(0.08) == STABLE
    assert classify_approach(-0.08) == STABLE
    assert classify_approach(0.0) == STABLE


def test_missing_values():
    assert classify_approach(None) == UNKNOWN
    assert classify_approach(math.nan) == UNKNOWN
    assert classify_approach(math.inf) == UNKNOWN
    assert classify_approach("invalid") == UNKNOWN


def test_threshold_boundary():
    assert classify_approach(0.15) == STABLE
    assert classify_approach(-0.15) == STABLE


def test_invalid_threshold():
    with pytest.raises(ValueError):
        classify_approach(0.3, threshold=0)

    with pytest.raises(ValueError):
        classify_approach(0.3, threshold=-0.1)
