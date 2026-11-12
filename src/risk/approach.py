
import math


APPROACHING = "APPROACHING"
RECEDING = "RECEDING"
STABLE = "STABLE"
UNKNOWN = "UNKNOWN"


def classify_approach(
    growth_rate,
    threshold=0.15,
):
    """
    Classify apparent vehicle-scale motion.

    Parameters
    ----------
    growth_rate : float or None
        Robust estimate of d(log(height))/dt,
        measured in inverse seconds.

    threshold : float
        Experimental deadband in inverse seconds.

    Returns
    -------
    str
        APPROACHING, RECEDING, STABLE, or UNKNOWN.

    Notes
    -----
    This is an image-space classification.
    It does not measure physical relative speed
    or establish collision risk.
    """

    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "threshold must be positive and finite"
        )

    if growth_rate is None:
        return UNKNOWN

    try:
        growth_rate = float(growth_rate)
    except (TypeError, ValueError):
        return UNKNOWN

    if not math.isfinite(growth_rate):
        return UNKNOWN

    if growth_rate > threshold:
        return APPROACHING

    if growth_rate < -threshold:
        return RECEDING

    return STABLE
