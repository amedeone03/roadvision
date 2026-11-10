
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
