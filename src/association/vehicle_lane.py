
import cv2
import numpy as np


def get_bottom_center(bbox):
    """
    Return the bottom-center of a bounding box.

    bbox format:
        (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = bbox

    x_center = (x1 + x2) / 2
    y_bottom = y2

    return (x_center, y_bottom)


def build_lane_polygon(left_fit, right_fit):
    """
    Construct a lane polygon from fitted boundaries.

    Each fit contains:
        (x_bottom, y_bottom, x_top, y_top)

    Returns:
        NumPy array of polygon vertices,
        or None if boundaries are unavailable.
    """

    if left_fit is None or right_fit is None:
        return None

    lb_x, lb_y, lt_x, lt_y = left_fit
    rb_x, rb_y, rt_x, rt_y = right_fit

    polygon = np.array(
        [
            [lb_x, lb_y],
            [lt_x, lt_y],
            [rt_x, rt_y],
            [rb_x, rb_y],
        ],
        dtype=np.float32,
    )

    return polygon


def is_vehicle_in_lane(bbox, lane_polygon):
    """
    Check whether the vehicle's bottom-center point
    lies inside or on the boundary of the ego lane.

    Returns:
        True  -> inside or on polygon boundary
        False -> outside polygon
        None  -> lane polygon unavailable
    """

    if lane_polygon is None:
        return None

    point = get_bottom_center(bbox)

    result = cv2.pointPolygonTest(
        lane_polygon,
        point,
        False,
    )

    return result >= 0


def classify_vehicle_lane(bbox, lane_polygon):
    """
    Classify vehicle position relative to the ego lane.

    Returns:
        IN_LANE  - bottom-center inside ego-lane polygon
        OUT_LANE - bottom-center outside polygon, but
                   within its supported vertical interval
        UNKNOWN  - lane unavailable or vehicle contact point
                   outside the supported vertical interval
    """

    if lane_polygon is None:
        return "UNKNOWN"

    x_center, y_bottom = get_bottom_center(bbox)

    y_min = float(lane_polygon[:, 1].min())
    y_max = float(lane_polygon[:, 1].max())

    if y_bottom < y_min or y_bottom > y_max:
        return "UNKNOWN"

    inside = cv2.pointPolygonTest(
        lane_polygon,
        (float(x_center), float(y_bottom)),
        False,
    )

    return "IN_LANE" if inside >= 0 else "OUT_LANE"


def build_extended_lane_polygon(
    left_fit,
    right_fit,
    y_top,
):
    """
    Extrapolate fitted lane boundaries to a new top Y.

    Returns None if extrapolated geometry is invalid.

    Important:
        This is an experimental image-space
        approximation, not a calibrated road model.
    """

    if left_fit is None or right_fit is None:
        return None

    lb_x, lb_y, lt_x, lt_y = left_fit
    rb_x, rb_y, rt_x, rt_y = right_fit

    if lb_y != rb_y or lt_y != rt_y:
        return None

    if lb_y == lt_y:
        return None

    if y_top >= lb_y or y_top < 0:
        return None

    # Fit x as a function of image height y.
    left_slope = (lt_x - lb_x) / (lt_y - lb_y)
    right_slope = (rt_x - rb_x) / (rt_y - rb_y)

    # Extrapolate both boundaries upward.
    left_top_x = lb_x + left_slope * (y_top - lb_y)
    right_top_x = rb_x + right_slope * (y_top - rb_y)

    # Avoid a crossed or collapsed polygon.
    if left_top_x >= right_top_x:
        return None

    polygon = np.array(
        [
            [lb_x, lb_y],
            [left_top_x, y_top],
            [right_top_x, y_top],
            [rb_x, rb_y],
        ],
        dtype=np.float32,
    )

    return polygon


def classify_vehicle_lane_detailed(bbox, lane_polygon):
    """
    Classify a vehicle relative to the ego lane.

    Returns:
        status:
            IN_LANE
            OUT_LANE
            UNKNOWN

        reason:
            inside_polygon
            outside_polygon
            lane_unavailable
            above_lane_coverage
            below_lane_coverage
    """

    # No valid ego-lane polygon is available.
    if lane_polygon is None:
        return "UNKNOWN", "lane_unavailable"

    # Approximate the vehicle's road-contact point.
    x_center, y_bottom = get_bottom_center(bbox)

    # Determine the supported vertical interval.
    y_min = float(lane_polygon[:, 1].min())
    y_max = float(lane_polygon[:, 1].max())

    # Vehicle is farther away than our lane model covers.
    if y_bottom < y_min:
        return "UNKNOWN", "above_lane_coverage"

    # Vehicle contact point is below the lane model.
    if y_bottom > y_max:
        return "UNKNOWN", "below_lane_coverage"

    # Test whether the point lies inside the polygon.
    result = cv2.pointPolygonTest(
        lane_polygon,
        (float(x_center), float(y_bottom)),
        False,
    )

    if result >= 0:
        return "IN_LANE", "inside_polygon"

    return "OUT_LANE", "outside_polygon"



def lane_x_at_y(fit, y):
    """
    Interpolate the X coordinate of a lane boundary
    at image height y.

    fit: (x_bottom, y_bottom, x_top, y_top)

    Returns None outside the fitted vertical interval.
    """

    if fit is None:
        return None

    x_bottom, y_bottom, x_top, y_top = fit

    if y_bottom == y_top:
        return None

    if not min(y_bottom, y_top) <= y <= max(
        y_bottom, y_top
    ):
        return None

    fraction = (y - y_top) / (y_bottom - y_top)

    return float(
        x_top + fraction * (x_bottom - x_top)
    )


def classify_vehicle_lane_conservative(
    bbox,
    left_fit,
    right_fit,
    margin=0.05,
):
    """
    Classify a tracked road user relative to
    the estimated ego lane.

    A margin of 0.05 reserves 5% of the estimated
    lane width near each boundary as uncertain.

    Returns:
        (status, reason, normalized_margin)

    A positive normalized margin means the contact
    point is inside the lateral lane boundaries.

    None indicates unavailable geometry or
    unsupported vertical coverage.
    """

    if not 0 <= margin < 0.5:
        raise ValueError(
            "margin must be between 0 and 0.5"
        )

    if left_fit is None or right_fit is None:
        return "UNKNOWN", "lane_unavailable", None

    x_center, y_bottom = get_bottom_center(bbox)

    # Check the common vertical coverage of both fits.
    y_min = max(left_fit[3], right_fit[3])
    y_max = min(left_fit[1], right_fit[1])

    if y_bottom < y_min:
        return "UNKNOWN", "above_lane_coverage", None

    if y_bottom > y_max:
        return "UNKNOWN", "below_lane_coverage", None

    x_left = lane_x_at_y(left_fit, y_bottom)
    x_right = lane_x_at_y(right_fit, y_bottom)

    if x_left is None or x_right is None:
        return "UNKNOWN", "invalid_lane_geometry", None

    lane_width = x_right - x_left

    if lane_width <= 0:
        return "UNKNOWN", "invalid_lane_geometry", None

    # Distances to the estimated lateral boundaries.
    d_left = x_center - x_left
    d_right = x_right - x_center

    normalized_margin = (
        min(d_left, d_right) / lane_width
    )

    # Clearly outside the lane.
    if d_left < 0 or d_right < 0:
        return (
            "OUT_LANE",
            "outside_lateral_boundaries",
            normalized_margin,
        )

    # Inside, but close to a boundary.
    if normalized_margin < margin:
        return (
            "UNKNOWN",
            "near_lane_boundary",
            normalized_margin,
        )

    return (
        "IN_LANE",
        "inside_lateral_boundaries",
        normalized_margin,
    )

