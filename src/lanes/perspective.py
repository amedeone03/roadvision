
import numpy as np


def estimate_vanishing_point(left_fit, right_fit):
    """
    Find the intersection of two fitted lane lines.

    Each line is represented as:
        (x_bottom, y_bottom, x_top, y_top)

    Returns:
        (x, y) intersection coordinates
        or None if the lines are parallel or invalid.
    """

    if left_fit is None or right_fit is None:
        return None

    def line_coefficients(fit):
        xb, yb, xt, yt = fit

        if yb == yt:
            return None

        # x = a*y + c
        a = (xt - xb) / (yt - yb)
        c = xb - a * yb

        return a, c

    left = line_coefficients(left_fit)
    right = line_coefficients(right_fit)

    if left is None or right is None:
        return None

    a_left, c_left = left
    a_right, c_right = right

    if np.isclose(a_left, a_right):
        return None

    y_intersection = (
        c_right - c_left
    ) / (
        a_left - a_right
    )

    x_intersection = (
        a_left * y_intersection + c_left
    )

    if not np.isfinite([x_intersection, y_intersection]).all():
        return None

    return float(x_intersection), float(y_intersection)
