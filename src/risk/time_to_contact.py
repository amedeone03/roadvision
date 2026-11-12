
import math


def estimate_tau_proxy(
    growth_rate,
    min_growth=0.15,
):
    """
    Estimate inverse apparent-scale growth time.

    tau = 1 / growth_rate

    Parameters
    ----------
    growth_rate : float or None
        Robust logarithmic bounding-box height
        growth, in inverse seconds.

    min_growth : float
        Minimum positive growth for an estimate.

    Returns
    -------
    float or None
        Experimental inverse-expansion time
        in seconds, or None when unavailable.

    Notes
    -----
    This is NOT a validated collision TTC.
    """

    if (
        not math.isfinite(min_growth)
        or min_growth <= 0
    ):
        raise ValueError(
            "min_growth must be positive and finite"
        )

    if growth_rate is None:
        return None

    try:
        growth = float(growth_rate)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(growth):
        return None

    if growth <= min_growth:
        return None

    return 1.0 / growth
