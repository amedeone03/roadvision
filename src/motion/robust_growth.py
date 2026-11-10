
import numpy as np


def theil_sen_slope(times, values):
    """
    Estimate a robust linear slope using the median
    of all pairwise slopes.

    Parameters
    ----------
    times : array-like
        Strictly increasing timestamps in seconds.

    values : array-like
        Measurements, e.g. logarithmic box heights.

    Returns
    -------
    float
        Estimated slope in measurement units / second.
    """

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if times.ndim != 1 or values.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(times) != len(values):
        raise ValueError("Inputs must have equal length")

    if len(times) < 2:
        raise ValueError("At least two observations required")

    if not (
        np.isfinite(times).all()
        and np.isfinite(values).all()
    ):
        raise ValueError("Inputs must contain finite values")

    if np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing")

    slopes = []

    for i in range(len(times) - 1):
        for j in range(i + 1, len(times)):

            slope = (
                (values[j] - values[i])
                / (times[j] - times[i])
            )

            slopes.append(slope)

    return float(np.median(slopes))


def rolling_robust_growth(times, heights, window=5):
    """
    Causal rolling Theil-Sen estimate of
    d(log(height)) / dt.

    Uses only the current and previous observations.

    Returns NaN until a complete window is available.

    Call separately for each consecutive tracking
    segment to avoid fitting across tracking gaps.
    """

    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")

    times = np.asarray(times, dtype=float)
    heights = np.asarray(heights, dtype=float)

    if times.ndim != 1 or heights.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(times) != len(heights):
        raise ValueError("Inputs must have equal length")

    if not (
        np.isfinite(times).all()
        and np.isfinite(heights).all()
    ):
        raise ValueError("Inputs must contain finite values")

    if np.any(heights <= 0):
        raise ValueError("Heights must be positive")

    if np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing")

    result = np.full(len(times), np.nan)

    log_heights = np.log(heights)

    for end in range(window - 1, len(times)):

        start = end - window + 1

        result[end] = theil_sen_slope(
            times[start:end + 1],
            log_heights[start:end + 1],
        )

    return result
