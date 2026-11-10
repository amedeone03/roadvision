
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.lanes.lane_detector import detect_lane
from src.lanes.vanishing_point import (
    estimate_vanishing_point,
    validate_vanishing_point,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

VIDEO_PATH = Path("data/videos/test_drive.mp4")

OUTPUT_DIR = Path(
    "runs/motion/automatic_vanishing_point"
)

OUTPUT_CSV = OUTPUT_DIR / "vp_results.csv"

# Manual measurements obtained from road markings.
# Coordinates refer to the original 1920x1080 video.
MANUAL_REFERENCES = {
    150: (877.50, 646.25),
    250: (886.22, 634.48),
    450: (871.78, 637.38),
    650: (974.91, 728.71),
    850: (867.61, 715.40),
    1023: (947.63, 634.21),
    1200: (915.76, 633.81),
    1450: (910.15, 640.16),
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# 2. LANE-FIT EXTRACTION
# ==================================================

def extract_lane_fits(lane_info):
    """
    Extract geometrically valid lane boundaries.

    RoadVision's detect_lane() returns:
        (annotated_frame, lane_info)

    lane_info contains:
        valid
        left_fit
        right_fit
        left_segments
        right_segments
        polygon

    Returns:
        (left_fit, right_fit)

    Invalid lane detections are excluded.
    """

    if not isinstance(lane_info, dict):
        raise TypeError(
            "Expected lane_info to be a dictionary. "
            f"Received {type(lane_info).__name__}."
        )

    if not lane_info.get("valid", False):
        return None, None

    left_fit = lane_info.get("left_fit")
    right_fit = lane_info.get("right_fit")

    if left_fit is None or right_fit is None:
        return None, None

    return left_fit, right_fit


# ==================================================
# 3. VISUALIZATION HELPERS
# ==================================================

def draw_extended_fit(frame, fit, color):
    """
    Draw a lane line extrapolated beyond the
    original fitting region.

    RoadVision fit format:
        (x_bottom, y_bottom, x_top, y_top)
    """

    if fit is None:
        return

    xb, yb, xt, yt = map(float, fit)

    if abs(yb - yt) < 1e-9:
        return

    slope = (xb - xt) / (yb - yt)

    def x_at_y(y):
        return int(round(
            xt + slope * (y - yt)
        ))

    height = frame.shape[0]

    y_start = int(0.35 * height)
    y_end = int(0.95 * height)

    cv2.line(
        frame,
        (x_at_y(y_start), y_start),
        (x_at_y(y_end), y_end),
        color,
        3,
    )


def draw_vanishing_point(
    frame,
    vp,
    color,
    label,
):
    """
    Draw an intersection and its horizontal row.
    """

    if vp is None:
        return

    x, y = vp

    if not (
        np.isfinite(x)
        and np.isfinite(y)
    ):
        return

    height, width = frame.shape[:2]

    if 0 <= y < height:
        y_int = int(round(y))

        cv2.line(
            frame,
            (0, y_int),
            (width - 1, y_int),
            color,
            2,
        )

    if (
        0 <= x < width
        and 0 <= y < height
    ):
        center = (
            int(round(x)),
            int(round(y)),
        )

        cv2.circle(
            frame,
            center,
            9,
            color,
            -1,
        )

        text_x = min(center[0] + 15, width - 250)
        text_y = max(center[1] - 15, 25)

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )


def save_comparison_image(
    frame,
    frame_number,
    left_fit,
    right_fit,
    raw_vp,
    reference,
    reason,
):
    """
    Save diagnostic overlays for manually
    annotated reference frames.
    """

    annotated = frame.copy()

    # Green: automatically detected left lane.
    draw_extended_fit(
        annotated,
        left_fit,
        (0, 255, 0),
    )

    # Blue: automatically detected right lane.
    draw_extended_fit(
        annotated,
        right_fit,
        (255, 0, 0),
    )

    # Orange: automatic raw intersection.
    draw_vanishing_point(
        annotated,
        raw_vp,
        (0, 165, 255),
        "AUTO",
    )

    # Magenta: manually estimated intersection.
    draw_vanishing_point(
        annotated,
        reference,
        (255, 0, 255),
        "MANUAL",
    )

    cv2.rectangle(
        annotated,
        (15, 15),
        (900, 100),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        annotated,
        f"Frame {frame_number}",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        annotated,
        f"VP status: {reason}",
        (30, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    output_path = (
        OUTPUT_DIR
        / f"compare_frame_{frame_number}.jpg"
    )

    if not cv2.imwrite(
        str(output_path),
        annotated,
    ):
        raise RuntimeError(
            f"Could not save {output_path}"
        )


# ==================================================
# 4. PROCESS VIDEO
# ==================================================

def evaluate_video():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("\n--- AUTOMATIC VANISHING POINT ---")
    print(f"Video: {VIDEO_PATH}")
    print(f"Frames: {total_frames}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")

    print(
        "Mode: raw geometrically valid lane fits"
    )

    print(
        "Temporal filtering: disabled "
        "for this experiment"
    )

    rows = []
    frame_number = 0
    valid_count = 0

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            # ----------------------------------
            # Run the existing Phase 3 detector.
            # ----------------------------------

            _, lane_info = detect_lane(frame)

            left_fit, right_fit = extract_lane_fits(
                lane_info
            )

            lane_geometry_valid = bool(
                lane_info.get("valid", False)
            )

            # ----------------------------------
            # Estimate line intersection.
            # ----------------------------------

            raw_vp = estimate_vanishing_point(
                left_fit,
                right_fit,
            )

            vp_valid, vp_reason = (
                validate_vanishing_point(
                    raw_vp,
                    width,
                    height,
                )
            )

            if vp_valid:
                valid_count += 1

            # Preserve raw intersections for
            # diagnosing rejected estimates.
            raw_x = (
                raw_vp[0]
                if raw_vp is not None
                else np.nan
            )

            raw_y = (
                raw_vp[1]
                if raw_vp is not None
                else np.nan
            )

            # Accepted automatic VP only.
            auto_x = (
                raw_x if vp_valid else np.nan
            )

            auto_y = (
                raw_y if vp_valid else np.nan
            )

            # ----------------------------------
            # Compare against manual references.
            # ----------------------------------

            reference = MANUAL_REFERENCES.get(
                frame_number
            )

            if reference is None:
                manual_x = np.nan
                manual_y = np.nan
            else:
                manual_x, manual_y = reference

            if vp_valid and reference is not None:

                error_x = auto_x - manual_x
                error_y = auto_y - manual_y

            else:
                error_x = np.nan
                error_y = np.nan

            rows.append({
                "frame": frame_number,

                "lane_geometry_valid":
                    lane_geometry_valid,

                "vp_raw_x": raw_x,
                "vp_raw_y": raw_y,

                "vp_x": auto_x,
                "vp_y": auto_y,

                "vp_valid": vp_valid,
                "vp_reason": vp_reason,

                "manual_x": manual_x,
                "manual_y": manual_y,

                "error_x": error_x,
                "error_y": error_y,

                "abs_error_x": abs(error_x),
                "abs_error_y": abs(error_y),
            })

            # ----------------------------------
            # Save manual-reference comparisons.
            # ----------------------------------

            if reference is not None:

                save_comparison_image(
                    frame=frame,
                    frame_number=frame_number,
                    left_fit=left_fit,
                    right_fit=right_fit,
                    raw_vp=raw_vp,
                    reference=reference,
                    reason=vp_reason,
                )

            # ----------------------------------
            # Progress updates.
            # ----------------------------------

            if frame_number % 200 == 0:

                print(
                    f"Frame {frame_number}/{total_frames} "
                    f"| Valid VP: {valid_count}"
                )

    finally:
        cap.release()

    # ==========================================
    # 5. EXPORT RESULTS
    # ==========================================

    if not rows:
        raise RuntimeError(
            "No video frames were processed"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ==========================================
    # 6. GLOBAL STATISTICS
    # ==========================================

    print("\n--- GLOBAL RESULTS ---")

    print(
        f"Frames processed: {len(df)}"
    )

    print(
        "Geometrically valid lane frames:",
        int(df["lane_geometry_valid"].sum()),
    )

    print(
        f"Valid VP estimates: {valid_count}"
    )

    print(
        "VP availability: "
        f"{100 * valid_count / len(df):.2f}%"
    )

    print("\nVP rejection reasons:")

    print(
        df["vp_reason"]
        .value_counts()
        .to_string()
    )

    # ==========================================
    # 7. MANUAL REFERENCE COMPARISON
    # ==========================================

    comparison = df[
        df["frame"].isin(
            MANUAL_REFERENCES.keys()
        )
    ].copy()

    print("\n--- MANUAL VS AUTOMATIC VP ---")

    print(
        comparison[
            [
                "frame",
                "manual_x",
                "manual_y",
                "vp_x",
                "vp_y",
                "abs_error_y",
                "vp_reason",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    matched = comparison[
        comparison["vp_valid"]
    ]

    print(
        "\nManual reference frames:",
        len(comparison),
    )

    print(
        "Matched valid reference frames:",
        len(matched),
    )

    if not matched.empty:

        print(
            "Median absolute X error: "
            f"{matched['abs_error_x'].median():.2f} px"
        )

        print(
            "Median absolute Y error: "
            f"{matched['abs_error_y'].median():.2f} px"
        )

        print(
            "Mean absolute Y error: "
            f"{matched['abs_error_y'].mean():.2f} px"
        )

        print(
            "Maximum absolute Y error: "
            f"{matched['abs_error_y'].max():.2f} px"
        )

    # ==========================================
    # 8. TEMPORAL STABILITY
    # ==========================================

    print("\n--- VP TEMPORAL STABILITY ---")

    previous_valid = (
        df["vp_valid"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    consecutive_valid = (
        df["vp_valid"] & previous_valid
    )

    vp_changes_y = (
        df["vp_y"].diff().abs()
    )

    valid_changes_y = (
        vp_changes_y[consecutive_valid]
        .dropna()
    )

    print(
        "Consecutive valid frame pairs:",
        len(valid_changes_y),
    )

    if not valid_changes_y.empty:

        print(
            "Median absolute Y change: "
            f"{valid_changes_y.median():.2f} px"
        )

        print(
            "95th percentile absolute Y change: "
            f"{valid_changes_y.quantile(0.95):.2f} px"
        )

        print(
            "Maximum absolute Y change: "
            f"{valid_changes_y.max():.2f} px"
        )

    # ==========================================
    # 9. VALID VP DISTRIBUTION
    # ==========================================

    valid_vps = df.loc[
        df["vp_valid"],
        ["vp_x", "vp_y"],
    ]

    print("\n--- VALID VP DISTRIBUTION ---")

    if not valid_vps.empty:

        print(
            valid_vps.describe(
                percentiles=[
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                ]
            )
            .round(2)
            .to_string()
        )

    # ==========================================
    # 10. FINAL OUTPUT
    # ==========================================

    print("\n--- OUTPUT FILES ---")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Images: {OUTPUT_DIR}")

    print(
        "\nAutomatic vanishing-point "
        "evaluation complete!"
    )


if __name__ == "__main__":
    evaluate_video()
