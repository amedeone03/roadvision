
import cv2
import numpy as np


# -----------------------------------------
# Configuration
# -----------------------------------------

BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150

MIN_ABS_SLOPE = 0.40
MAX_ABS_SLOPE = 3.0

HOUGH_THRESHOLD = 40
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 30


def fit_lane_line(segments, y_bottom, y_top):
    """Fit x = a*y + c using length-weighted segments."""

    if not segments:
        return None

    x_points = []
    y_points = []
    weights = []

    for x1, y1, x2, y2 in segments:
        length = np.hypot(x2 - x1, y2 - y1)

        if length < 1:
            continue

        x_points.extend([x1, x2])
        y_points.extend([y1, y2])
        weights.extend([length, length])

    if len(x_points) < 2:
        return None

    # Avoid fitting degenerate vertical-coordinate data
    if np.ptp(y_points) < 1:
        return None

    try:
        a, c = np.polyfit(
            y_points,
            x_points,
            deg=1,
            w=np.sqrt(weights),
        )
    except (np.linalg.LinAlgError, ValueError):
        return None

    if not np.isfinite([a, c]).all():
        return None

    return (
        int(a * y_bottom + c),
        y_bottom,
        int(a * y_top + c),
        y_top,
    )



def validate_lane_polygon(left_fit, right_fit, image_width):
    """
    Validate the estimated ego-lane geometry.

    Width thresholds are experimental values
    calibrated for the current 1920x1080 footage.
    """

    # Both lane boundaries must exist
    if left_fit is None or right_fit is None:
        return False

    lb_x, lb_y, lt_x, lt_y = left_fit
    rb_x, rb_y, rt_x, rt_y = right_fit

    # 1. Both boundaries must use the same
    # vertical interval
    if (
        lb_y != rb_y
        or lt_y != rt_y
        or lt_y >= lb_y
    ):
        return False

    # 2. Calculate lane widths
    bottom_width = rb_x - lb_x
    top_width = rt_x - lt_x

    # 3. Left boundary must remain to the left
    # of the right boundary
    if bottom_width <= 0 or top_width <= 0:
        return False

    # 4. Endpoints must be inside the image
    if any(
        x < 0 or x >= image_width
        for x in (lb_x, lt_x, rb_x, rt_x)
    ):
        return False

    # 5. Width constraints, normalized to
    # the image width (reference: 1920 px)
    scale = image_width / 1920.0

    MIN_BOTTOM_WIDTH = 650 * scale
    MAX_BOTTOM_WIDTH = 1350 * scale

    MIN_TOP_WIDTH = 0
    MAX_TOP_WIDTH = 500 * scale

    if not (
        MIN_BOTTOM_WIDTH
        <= bottom_width
        <= MAX_BOTTOM_WIDTH
    ):
        return False

    if not (
        MIN_TOP_WIDTH
        < top_width
        <= MAX_TOP_WIDTH
    ):
        return False

    # 6. Perspective consistency:
    # the lane should appear wider at the bottom
    if bottom_width <= top_width:
        return False

    return True


def detect_lane(frame):
    """
    Detect approximately straight ego-lane boundaries.

    Returns:
        annotated_frame: image with lane overlay
        lane_info: dictionary containing detection information
    """

    height, width = frame.shape[:2]

    # -------------------------------------
    # 1. Image preprocessing
    # -------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray, BLUR_KERNEL, 0
    )

    edges = cv2.Canny(
        blurred, CANNY_LOW, CANNY_HIGH
    )

    # -------------------------------------
    # 2. Region of interest
    # -------------------------------------

    vertices = np.array([[
        (int(0.20 * width), height - 1),
        (int(0.43 * width), int(0.52 * height)),
        (int(0.58 * width), int(0.52 * height)),
        (int(0.95 * width), height - 1),
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)

    masked_edges = cv2.bitwise_and(edges, mask)

    # -------------------------------------
    # 3. Hough line detection
    # -------------------------------------

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    # -------------------------------------
    # 4. Filter candidate segments
    # -------------------------------------

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / dx

            if not (
                MIN_ABS_SLOPE
                <= abs(slope)
                <= MAX_ABS_SLOPE
            ):
                continue

            mid_x = (x1 + x2) / 2

            if slope < 0 and mid_x < width / 2:
                left_lines.append((x1, y1, x2, y2))

            elif slope > 0 and mid_x > width / 2:
                right_lines.append((x1, y1, x2, y2))

    # -------------------------------------
    # 5. Fit lane boundaries
    # -------------------------------------

    y_bottom = int(0.90 * height)
    y_top = int(0.68 * height)

    left_fit = fit_lane_line(
        left_lines, y_bottom, y_top
    )

    right_fit = fit_lane_line(
        right_lines, y_bottom, y_top
    )

    # -------------------------------------
    # 6. Validate lane geometry
    # -------------------------------------

    valid = validate_lane_polygon(
        left_fit, right_fit, width
    )

    annotated = frame.copy()

    lane_info = {
        "valid": valid,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "left_segments": len(left_lines),
        "right_segments": len(right_lines),
    }

    if not valid:
        return annotated, lane_info

    # -------------------------------------
    # 7. Construct ego-lane polygon
    # -------------------------------------

    lb_x, lb_y, lt_x, lt_y = left_fit
    rb_x, rb_y, rt_x, rt_y = right_fit

    polygon = np.array([[
        [lb_x, lb_y],
        [lt_x, lt_y],
        [rt_x, rt_y],
        [rb_x, rb_y],
    ]], dtype=np.int32)

    lane_info["polygon"] = polygon[0].tolist()

    # -------------------------------------
    # 8. Draw overlay
    # -------------------------------------

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        polygon,
        (0, 255, 0)
    )

    annotated = cv2.addWeighted(
        overlay, 0.25,
        frame, 0.75,
        0
    )

    cv2.line(
        annotated,
        (lb_x, lb_y),
        (lt_x, lt_y),
        (0, 255, 0),
        6
    )

    cv2.line(
        annotated,
        (rb_x, rb_y),
        (rt_x, rt_y),
        (255, 0, 0),
        6
    )

    return annotated, lane_info
