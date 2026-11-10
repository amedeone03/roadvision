
from pathlib import Path
from unittest.mock import patch

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
TRACKING_CSV = Path("runs/integrated_final.csv")

OUTPUT_DIR = Path("runs/motion/full_video_vp")
OUTPUT_CSV = OUTPUT_DIR / "full_video_vp_results.csv"

BOX_PADDING = 0.03
INPAINT_RADIUS = 5

METHODS = [
    "ORIGINAL",
    "INPAINTED",
    "EDGE_MASKED",
]

REFERENCE_FRAMES = {
    150: (877.50, 646.25),
    250: (886.22, 634.48),
    450: (871.78, 637.38),
    650: (974.91, 728.71),
    850: (867.61, 715.40),
    1023: (947.63, 634.21),
    1200: (915.76, 633.81),
    1450: (910.15, 640.16),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD TRACKING DATA
# ==================================================

if not TRACKING_CSV.exists():
    raise FileNotFoundError(TRACKING_CSV)

tracks = pd.read_csv(TRACKING_CSV)

required = {
    "frame", "x1", "y1", "x2", "y2"
}

missing = required - set(tracks.columns)

if missing:
    raise ValueError(
        f"Missing tracking columns: {sorted(missing)}"
    )

tracks_by_frame = {
    int(frame): group
    for frame, group in tracks.groupby("frame")
}

EMPTY_TRACKS = tracks.iloc[0:0]


# ==================================================
# 3. OBJECT MASK
# ==================================================

def make_object_mask(frame, observations):

    height, width = frame.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for row in observations.itertuples(index=False):

        x1 = float(row.x1)
        y1 = float(row.y1)
        x2 = float(row.x2)
        y2 = float(row.y2)

        if not np.isfinite(
            [x1, y1, x2, y2]
        ).all():
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        pad_x = BOX_PADDING * (x2 - x1)
        pad_y = BOX_PADDING * (y2 - y1)

        left = max(0, int(np.floor(x1 - pad_x)))
        top = max(0, int(np.floor(y1 - pad_y)))

        right = min(
            width - 1,
            int(np.ceil(x2 + pad_x)),
        )

        bottom = min(
            height - 1,
            int(np.ceil(y2 + pad_y)),
        )

        if right <= left or bottom <= top:
            continue

        cv2.rectangle(
            mask,
            (left, top),
            (right, bottom),
            255,
            -1,
        )

    return mask


# ==================================================
# 4. VP ESTIMATION
# ==================================================

def evaluate_lane(frame):
    """
    Run the existing lane detector and estimate
    the intersection of its valid fitted lines.
    """

    _, lane_info = detect_lane(frame)

    if not isinstance(lane_info, dict):
        raise TypeError(
            "Unexpected lane detector output"
        )

    lane_valid = bool(
        lane_info.get("valid", False)
    )

    left_fit = (
        lane_info.get("left_fit")
        if lane_valid else None
    )

    right_fit = (
        lane_info.get("right_fit")
        if lane_valid else None
    )

    vp = estimate_vanishing_point(
        left_fit,
        right_fit,
    )

    height, width = frame.shape[:2]

    valid, reason = validate_vanishing_point(
        vp,
        width,
        height,
    )

    return {
        "vp": vp,
        "valid": valid,
        "reason": reason,
        "lane_valid": lane_valid,
        "left_fit": left_fit,
        "right_fit": right_fit,
    }


def evaluate_edge_masked(frame, mask):
    """
    Experimental interception of HoughLinesP.

    Mask pixels are removed from the edge map,
    while the original RGB frame is preserved.

    This is not intended for parallel execution.
    """

    original_hough = cv2.HoughLinesP
    calls = 0
    removed_edges = 0

    def masked_hough(edges, *args, **kwargs):

        nonlocal calls, removed_edges

        calls += 1

        if edges.shape[:2] != mask.shape:
            raise RuntimeError(
                "Mask and edge dimensions differ"
            )

        filtered = edges.copy()

        removed_edges += int(
            np.count_nonzero(filtered[mask > 0])
        )

        filtered[mask > 0] = 0

        return original_hough(
            filtered, *args, **kwargs
        )

    with patch.object(
        cv2,
        "HoughLinesP",
        side_effect=masked_hough,
    ):
        result = evaluate_lane(frame)

    if calls == 0:
        raise RuntimeError(
            "HoughLinesP was not intercepted. "
            "Check the Phase 3 implementation."
        )

    result["removed_edges"] = removed_edges

    return result


# ==================================================
# 5. DRAWING
# ==================================================

def draw_result(frame, result, reference=None):

    image = frame.copy()
    height, width = image.shape[:2]

    for fit, color in [
        (result["left_fit"], (0, 255, 0)),
        (result["right_fit"], (255, 0, 0)),
    ]:

        if fit is None:
            continue

        xb, yb, xt, yt = map(float, fit)

        if abs(yb - yt) < 1e-9:
            continue

        slope = (xb - xt) / (yb - yt)

        def at_y(y):
            x = xt + slope * (y - yt)
            return int(round(x)), int(round(y))

        cv2.line(
            image,
            at_y(0.35 * height),
            at_y(0.95 * height),
            color,
            3,
        )

    for vp, color in [
        (result["vp"], (0, 165, 255)),
        (reference, (255, 0, 255)),
    ]:

        if vp is None:
            continue

        x, y = vp

        if (
            np.isfinite([x, y]).all()
            and 0 <= x < width
            and 0 <= y < height
        ):
            cv2.circle(
                image,
                (int(round(x)), int(round(y))),
                10,
                color,
                -1,
            )

    return cv2.resize(image, (640, 360))


# ==================================================
# 6. PROCESS FULL VIDEO
# ==================================================

def main():

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("\n--- FULL-VIDEO VP EVALUATION ---")
    print(f"Total frames: {total_frames}")
    print("Methods: ORIGINAL, INPAINTED, EDGE_MASKED")

    rows = []
    frame_number = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            observations = tracks_by_frame.get(
                frame_number,
                EMPTY_TRACKS,
            )

            mask = make_object_mask(
                frame,
                observations,
            )

            # ------------------------------
            # ORIGINAL
            # ------------------------------

            original = evaluate_lane(frame)

            # ------------------------------
            # RGB INPAINTING
            # ------------------------------

            if np.any(mask):
                inpainted_frame = cv2.inpaint(
                    frame,
                    mask,
                    INPAINT_RADIUS,
                    cv2.INPAINT_TELEA,
                )
            else:
                inpainted_frame = frame.copy()

            inpainted = evaluate_lane(
                inpainted_frame
            )

            # ------------------------------
            # EDGE MASKING
            # ------------------------------

            edge_masked = evaluate_edge_masked(
                frame,
                mask,
            )

            results = {
                "ORIGINAL": original,
                "INPAINTED": inpainted,
                "EDGE_MASKED": edge_masked,
            }

            manual = REFERENCE_FRAMES.get(
                frame_number
            )

            for method, result in results.items():

                vp = result["vp"]
                valid = bool(result["valid"])

                raw_x = (
                    float(vp[0])
                    if vp is not None
                    else np.nan
                )

                raw_y = (
                    float(vp[1])
                    if vp is not None
                    else np.nan
                )

                error_y = (
                    abs(raw_y - manual[1])
                    if valid and manual is not None
                    else np.nan
                )

                rows.append({
                    "frame": frame_number,
                    "method": method,

                    "lane_valid":
                        result["lane_valid"],

                    "vp_valid": valid,
                    "vp_reason": result["reason"],

                    "vp_x": (
                        raw_x if valid else np.nan
                    ),

                    "vp_y": (
                        raw_y if valid else np.nan
                    ),

                    "vp_raw_x": raw_x,
                    "vp_raw_y": raw_y,

                    "tracked_objects": len(
                        observations
                    ),

                    "mask_fraction": (
                        np.count_nonzero(mask)
                        / mask.size
                    ),

                    "removed_edge_pixels": (
                        result.get(
                            "removed_edges",
                            np.nan,
                        )
                    ),

                    "manual_y": (
                        manual[1]
                        if manual is not None
                        else np.nan
                    ),

                    "abs_error_y": error_y,
                })

            # Save diagnostic comparisons.
            if frame_number in REFERENCE_FRAMES:

                reference = REFERENCE_FRAMES[
                    frame_number
                ]

                panels = [
                    draw_result(
                        frame,
                        original,
                        reference,
                    ),
                    draw_result(
                        inpainted_frame,
                        inpainted,
                        reference,
                    ),
                    draw_result(
                        frame,
                        edge_masked,
                        reference,
                    ),
                ]

                combined = np.hstack(panels)

                cv2.imwrite(
                    str(
                        OUTPUT_DIR
                        / f"compare_{frame_number}.jpg"
                    ),
                    combined,
                )

            if frame_number % 200 == 0:

                print(
                    f"Processed "
                    f"{frame_number}/{total_frames}"
                )

    finally:
        cap.release()

    if not rows:
        raise RuntimeError(
            "No frames were processed"
        )

    df = pd.DataFrame(rows)

    # ==================================================
    # 7. TEMPORAL METRICS
    # ==================================================

    df = df.sort_values(
        ["method", "frame"]
    ).reset_index(drop=True)

    df["previous_frame"] = (
        df.groupby("method")["frame"].shift(1)
    )

    df["previous_vp_y"] = (
        df.groupby("method")["vp_y"].shift(1)
    )

    consecutive = (
        df["frame"] - df["previous_frame"] == 1
    )

    df["abs_delta_vp_y"] = (
        df["vp_y"] - df["previous_vp_y"]
    ).abs().where(
        consecutive
        & df["vp_y"].notna()
        & df["previous_vp_y"].notna()
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ==================================================
    # 8. SUMMARY
    # ==================================================

    print("\n--- GLOBAL COMPARISON ---")

    for method in METHODS:

        subset = df[
            df["method"] == method
        ]

        valid = subset[
            subset["vp_valid"]
        ]

        changes = subset[
            "abs_delta_vp_y"
        ].dropna()

        print(f"\n{method}")

        print(
            f"Valid: {len(valid)}/{len(subset)} "
            f"({len(valid) / len(subset):.2%})"
        )

        print(
            "Consecutive valid pairs:",
            len(changes),
        )

        if not changes.empty:

            print(
                "Median absolute VP change: "
                f"{changes.median():.2f} px"
            )

            print(
                "95th percentile VP change: "
                f"{changes.quantile(0.95):.2f} px"
            )

            print(
                "Jumps greater than 20 px:",
                int((changes > 20).sum()),
            )

            print(
                "Jumps greater than 50 px:",
                int((changes > 50).sum()),
            )

        print(
            "Rejection reasons:"
        )

        print(
            subset["vp_reason"]
            .value_counts()
            .to_string()
        )

    # ==================================================
    # 9. MANUAL REFERENCE COMPARISON
    # ==================================================

    print("\n--- REFERENCE FRAMES ---")

    comparison = df[
        df["frame"].isin(REFERENCE_FRAMES)
    ]

    errors = comparison.pivot(
        index="frame",
        columns="method",
        values="abs_error_y",
    )

    print(
        errors.round(2).to_string()
    )

    print("\n--- OUTPUT ---")
    print(OUTPUT_CSV)
    print(OUTPUT_DIR)
    print("\nFull-video evaluation complete!")


if __name__ == "__main__":
    main()
