
import math


def estimate_vanishing_point(left_fit, right_fit):
    """
    Estimate the intersection of two infinite lane lines.

    Each fit uses the RoadVision format:
        (x_bottom, y_bottom, x_top, y_top)

    Returns:
        (vp_x, vp_y), or None if geometry is invalid.

    The result is an extrapolated line intersection,
    not a complete camera calibration.
    """

    if left_fit is None or right_fit is None:
        return None

    def line_parameters(fit):
        if len(fit) != 4:
            raise ValueError(
                "Lane fit must contain four coordinates"
            )

        xb, yb, xt, yt = map(float, fit)

        if not all(
            math.isfinite(v)
            for v in (xb, yb, xt, yt)
        ):
            return None

        if abs(yb - yt) < 1e-9:
            return None

        # x = a*y + b
        a = (xb - xt) / (yb - yt)
        b = xb - a * yb

        return a, b

    left = line_parameters(left_fit)
    right = line_parameters(right_fit)

    if left is None or right is None:
        return None

    a_left, b_left = left
    a_right, b_right = right

    denominator = a_left - a_right

    # Nearly parallel image-space lines lead to
    # an unstable intersection.
    if abs(denominator) < 1e-6:
        return None

    vp_y = (b_right - b_left) / denominator
    vp_x = a_left * vp_y + b_left

    if not (
        math.isfinite(vp_x)
        and math.isfinite(vp_y)
    ):
        return None

    return float(vp_x), float(vp_y)


def validate_vanishing_point(
    vp,
    image_width,
    image_height,
    min_y_fraction=0.35,
    max_y_fraction=0.80,
):
    """
    Broad geometric plausibility check.

    These thresholds are diagnostic, not
    calibrated vanishing-point confidence bounds.
    """

    if vp is None:
        return False, "missing_intersection"

    x, y = vp

    if not (0 <= x < image_width):
        return False, "vp_outside_image_x"

    if not (
        min_y_fraction * image_height
        <= y
        <= max_y_fraction * image_height
    ):
        return False, "vp_outside_expected_y_region"

    return True, "valid"
