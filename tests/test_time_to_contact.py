
import math

import pytest

from src.risk.time_to_contact import (
    estimate_tau_proxy,
)


def test_positive_growth():
    assert estimate_tau_proxy(0.5) == pytest.approx(2.0)
    assert estimate_tau_proxy(0.25) == pytest.approx(4.0)
    assert estimate_tau_proxy(1.0) == pytest.approx(1.0)


def test_nonapproaching():
    assert estimate_tau_proxy(0.0) is None
    assert estimate_tau_proxy(-0.5) is None
    assert estimate_tau_proxy(0.1) is None


def test_threshold_boundary():
    assert estimate_tau_proxy(0.15) is None
    assert estimate_tau_proxy(0.1501) is not None


def test_missing_or_invalid_values():
    for value in [None, math.nan, math.inf, -math.inf, "bad"]:
        assert estimate_tau_proxy(value) is None


def test_custom_threshold():
    assert estimate_tau_proxy(
        0.2, min_growth=0.1
    ) == pytest.approx(5.0)

    assert estimate_tau_proxy(
        0.2, min_growth=0.3
    ) is None


def test_invalid_threshold():
    for threshold in [0.0, -0.1, math.nan, math.inf]:
        with pytest.raises(ValueError):
            estimate_tau_proxy(
                0.5, min_growth=threshold
            )
