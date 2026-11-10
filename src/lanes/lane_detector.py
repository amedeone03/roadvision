
import cv2
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    RANSACRegressor,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BLUR_KERNEL = (5, 5)

CANNY_LOW = 50
CANNY_HIGH = 150

MIN_ABS_SLOPE = 0.40
MAX_ABS_SLOPE = 3.0

HOUGH_THRESHOLD = 40
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 30


# --------------------------------------------------
# 2. Weighted least-squares fitting
# --------------------------------------------------

def fit_lane_line(segments, y_bottom, y_top):
    """
    Fit x = a*y + c using length-weighted
    Hough segment endpoints.
    """

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


# --------------------------------------------------
# 3. RANSAC fitting
# --------------------------------------------------

def fit_lane_line_ransac(
    segments,
    y_bottom,
    y_top,
    residual_threshold=15,
):
    """
    Fit x = a*y + c using RANSAC.

    Each Hough segment contributes two endpoints.

    Returns:
        (x_bottom, y_bottom, x_top, y_top)
        or None.
    """

    if not segments:
        return None

    x_points = []
    y_points = []

    for x1, y1, x2, y2 in segments:

        x_points.extend([x1, x2])
        y_points.extend([y1, y2])

    if len(x_points) < 4:
        return None

    X = np.array(
        y_points,
        dtype=float,
    ).reshape(-1, 1)

    y = np.array(
        x_points,
        dtype=float,
    )

    if np.ptp(X) < 1:
        return None

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=2,
        residual_threshold=residual_threshold,
        max_trials=100,
        random_state=42,
    )

    try:
        ransac.fit(X, y)

        predicted = ransac.predict(
            np.array(
                [[y_bottom], [y_top]],
                dtype=float,
            )
        )

    except (ValueError, np.linalg.LinAlgError):
        return None

    if not np.isfinite(predicted).all():
        return None

    return (
        int(predicted[0]),
        y_bottom,
        int(predicted[1]),
        y_top,
    )


# --------------------------------------------------
# 4. Geometric validation
# --------------------------------------------------

def validate_lane_polygon(
    left_fit,
    right_fit,
    image_width,
):
    """
    Experimental geometry validation for
    the current dashcam recording.
    """

    if left_fit is None or right_fit is None:
        return False

    lb_x, lb_y, lt_x, lt_y = left_fit
    rb_x, rb_y, rt_x, rt_y = right_fit

    if (
        lb_y != rb_y
        or lt_y != rt_y
        or lt_y >= lb_y
    ):
        return False

    bottom_width = rb_x - lb_x
    top_width = rt_x - lt_x

    if bottom_width <= 0 or top_width <= 0:
        return False

    if any(
        x < 0 or x >= image_width
        for x in (lb_x, lt_x, rb_x, rt_x)
    ):
        return False

    scale = image_width / 1920.0

    min_bottom_width = 650 * scale
    max_bottom_width = 1350 * scale

    min_top_width = 0
    max_top_width = 500 * scale

    if not (
        min_bottom_width
        <= bottom_width
        <= max_bottom_width
    ):
        return False

    if not (
        min_top_width
        < top_width
        <= max_top_width
    ):
        return False

    if bottom_width <= top_width:
        return False

    return True


# --------------------------------------------------
# 5. Main lane detection pipeline
# --------------------------------------------------

def detect_lane(frame, return_debug=False):
    """
    Detect approximately straight ego-lane boundaries.

    Weighted least squares remains the main detector.
    RANSAC is computed as a diagnostic alternative.

    Default:
        annotated_frame, lane_info

    When return_debug=True:
        annotated_frame, lane_info, debug_frame
    """

    height, width = frame.shape[:2]

    # ------------------------------------------
    # 5.1 Grayscale, blur, Canny
    # ------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        BLUR_KERNEL,
        0,
    )

    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH,
    )

    # ------------------------------------------
    # 5.2 Expanded experimental ROI
    # ------------------------------------------

    vertices = np.array([[
        (int(0.20 * width), height - 1),
        (int(0.38 * width), int(0.48 * height)),
        (int(0.62 * width), int(0.48 * height)),
        (int(0.95 * width), height - 1),
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)

    cv2.fillPoly(mask, vertices, 255)

    masked_edges = cv2.bitwise_and(
        edges,
        mask,
    )

    # ------------------------------------------
    # 5.3 Probabilistic Hough transform
    # ------------------------------------------

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    # ------------------------------------------
    # 5.4 Candidate filtering
    # ------------------------------------------

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

                left_lines.append(
                    (x1, y1, x2, y2)
                )

            elif slope > 0 and mid_x > width / 2:

                right_lines.append(
                    (x1, y1, x2, y2)
                )

    # ------------------------------------------
    # 5.5 Fit both methods
    # ------------------------------------------

    y_bottom = int(0.90 * height)
    y_top = int(0.55 * height)

    # Existing weighted least-squares fit
    left_fit = fit_lane_line(
        left_lines,
        y_bottom,
        y_top,
    )

    right_fit = fit_lane_line(
        right_lines,
        y_bottom,
        y_top,
    )

    # Experimental RANSAC fit
    left_ransac = fit_lane_line_ransac(
        left_lines,
        y_bottom,
        y_top,
    )

    right_ransac = fit_lane_line_ransac(
        right_lines,
        y_bottom,
        y_top,
    )

    # ------------------------------------------
    # 5.6 Validate both fitting methods
    # ------------------------------------------

    valid = validate_lane_polygon(
        left_fit,
        right_fit,
        width,
    )

    ransac_valid = validate_lane_polygon(
        left_ransac,
        right_ransac,
        width,
    )

    lane_info = {
        "valid": valid,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "left_segments": len(left_lines),
        "right_segments": len(right_lines),
        "left_ransac": left_ransac,
        "right_ransac": right_ransac,
        "ransac_valid": ransac_valid,
    }

    # ------------------------------------------
    # 5.7 Optional debug visualization
    # ------------------------------------------

    debug_frame = None

    if return_debug:

        debug_frame = frame.copy()

        # ROI in yellow
        cv2.polylines(
            debug_frame,
            vertices,
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )

        # Candidate left segments: green
        for x1, y1, x2, y2 in left_lines:

            cv2.line(
                debug_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4,
            )

        # Candidate right segments: blue
        for x1, y1, x2, y2 in right_lines:

            cv2.line(
                debug_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                4,
            )

        # Weighted fits: thin green and blue
        if left_fit is not None:

            cv2.line(
                debug_frame,
                (left_fit[0], left_fit[1]),
                (left_fit[2], left_fit[3]),
                (0, 150, 0),
                2,
            )

        if right_fit is not None:

            cv2.line(
                debug_frame,
                (right_fit[0], right_fit[1]),
                (right_fit[2], right_fit[3]),
                (150, 0, 0),
                2,
            )

        # RANSAC fits: thick magenta and cyan
        if left_ransac is not None:

            cv2.line(
                debug_frame,
                (left_ransac[0], left_ransac[1]),
                (left_ransac[2], left_ransac[3]),
                (255, 0, 255),
                5,
            )

        if right_ransac is not None:

            cv2.line(
                debug_frame,
                (right_ransac[0], right_ransac[1]),
                (right_ransac[2], right_ransac[3]),
                (255, 255, 0),
                5,
            )

        cv2.putText(
            debug_frame,
            (
                f"Weighted: {'VALID' if valid else 'REJECTED'}"
                f" | RANSAC: {'VALID' if ransac_valid else 'REJECTED'}"
            ),
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    # ------------------------------------------
    # 5.8 Main output: weighted regression
    # ------------------------------------------

    annotated = frame.copy()

    if valid:

        lb_x, lb_y, lt_x, lt_y = left_fit
        rb_x, rb_y, rt_x, rt_y = right_fit

        polygon = np.array([[
            [lb_x, lb_y],
            [lt_x, lt_y],
            [rt_x, rt_y],
            [rb_x, rb_y],
        ]], dtype=np.int32)

        lane_info["polygon"] = polygon[0].tolist()

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            polygon,
            (0, 255, 0),
        )

        annotated = cv2.addWeighted(
            overlay,
            0.25,
            frame,
            0.75,
            0,
        )

        cv2.line(
            annotated,
            (lb_x, lb_y),
            (lt_x, lt_y),
            (0, 255, 0),
            6,
        )

        cv2.line(
            annotated,
            (rb_x, rb_y),
            (rt_x, rt_y),
            (255, 0, 0),
            6,
        )

    # ------------------------------------------
    # 5.9 Return
    # ------------------------------------------

    if return_debug:
        return annotated, lane_info, debug_frame

    return annotated, lane_info
