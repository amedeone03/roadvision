
import math


def bounding_box_features(bbox):
    """
    Extract geometric features from a bounding box.

    bbox:
        (x1, y1, x2, y2)

    Returns a dictionary containing:
        center_x, center_y,
        width, height,
        area,
        contact_x, contact_y
    """

    x1, y1, x2, y2 = map(float, bbox)

    values = (x1, y1, x2, y2)

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Bounding box contains non-finite values")

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        raise ValueError("Bounding box dimensions must be positive")

    return {
        "center_x": (x1 + x2) / 2.0,
        "center_y": (y1 + y2) / 2.0,
        "width": width,
        "height": height,
        "area": width * height,
        "contact_x": (x1 + x2) / 2.0,
        "contact_y": y2,
    }


def calculate_image_motion(previous, current, dt):
    """
    Calculate image-space motion between two
    consecutive observations of the same track.

    previous/current:
        Dictionaries from bounding_box_features()

    dt:
        Elapsed time in seconds.

    Returns:
        velocity_x_px_s
        velocity_y_px_s
        contact_velocity_x_px_s
        contact_velocity_y_px_s
        height_growth_px_s
        log_height_growth_s
        area_growth_px2_s
    """

    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    velocity_x = (
        current["center_x"] - previous["center_x"]
    ) / dt

    velocity_y = (
        current["center_y"] - previous["center_y"]
    ) / dt

    contact_velocity_x = (
        current["contact_x"] - previous["contact_x"]
    ) / dt

    contact_velocity_y = (
        current["contact_y"] - previous["contact_y"]
    ) / dt

    height_growth = (
        current["height"] - previous["height"]
    ) / dt

    log_height_growth = (
        math.log(current["height"])
        - math.log(previous["height"])
    ) / dt

    area_growth = (
        current["area"] - previous["area"]
    ) / dt

    return {
        "velocity_x_px_s": velocity_x,
        "velocity_y_px_s": velocity_y,
        "contact_velocity_x_px_s": contact_velocity_x,
        "contact_velocity_y_px_s": contact_velocity_y,
        "height_growth_px_s": height_growth,
        "log_height_growth_s": log_height_growth,
        "area_growth_px2_s": area_growth,
    }
