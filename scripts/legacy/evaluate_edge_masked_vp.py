
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

OUTPUT_DIR = Path("runs/motion/edge_masked_vp")
OUTPUT_CSV = OUTPUT_DIR / "comparison.csv"

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

BOX_PADDING = 0.03
INPAINT_RADIUS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD TRACKING DATA
# ==================================================

tracks = pd.read_csv(TRACKING_CSV)

required_columns = {
    "frame", "track_id",
    "x1", "y1", "x2", "y2",
}

missing = required_columns - set(tracks.columns)

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

tracks_by_frame = {
    int(frame_number): group
    for frame_number, group in tracks.groupby("frame")
}


# ==================================================
# 3. BUILD OBJECT MASK
# ==================================================

def make_object_mask(frame, observations):

    height, width = frame.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for row in observations.itertuples(index=False):

        x1, y1, x2, y2 = (
            float(row.x1),
            float(row.y1),
            float(row.x2),
            float(row.y2),
        )

        values = [x1, y1, x2, y2]

        if not np.isfinite(values).all():
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        pad_x = BOX_PADDING * (x2 - x1)
        pad_y = BOX_PADDING * (y2 - y1)

        left = max(
            0, int(np.floor(x1 - pad_x))
        )
        top = max(
            0, int(np.floor(y1 - pad_y))
        )
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
# 4. SHARED VP EVALUATION
# ==================================================

def evaluate_lane_geometry(frame):

    _, lane_info = detect_lane(frame)

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
        "lane_valid": lane_valid,
        "vp": vp,
        "vp_valid": valid,
        "vp_reason": reason,
        "left_fit": left_fit,
        "right_fit": right_fit,
    }


# ==================================================
# 5. EDGE-MASKED DETECTOR
# ==================================================

def evaluate_edge_masked(frame, object_mask):
    """
    Reuse detect_lane() but remove object pixels
    from the edge map passed to cv2.HoughLinesP.

    Intended for a single-threaded experiment.

    This relies on detect_lane() calling
    cv2.HoughLinesP rather than importing
    HoughLinesP directly.
    """

    original_hough = cv2.HoughLinesP

    calls = {"count": 0}
    masked_edge_pixels = {"count": 0}

    def masked_hough(edges, *args, **kwargs):

        calls["count"] += 1

        if edges.shape[:2] != object_mask.shape:
            raise ValueError(
                "Edge map and object mask dimensions "
                "do not match"
            )

        filtered_edges = edges.copy()

        masked_edge_pixels["count"] += int(
            np.count_nonzero(
                filtered_edges[object_mask > 0]
            )
        )

        filtered_edges[object_mask > 0] = 0

        return original_hough(
            filtered_edges,
            *args,
            **kwargs,
        )

    with patch.object(
        cv2,
        "HoughLinesP",
        side_effect=masked_hough,
    ):
        result = evaluate_lane_geometry(frame)

    if calls["count"] == 0:
        raise RuntimeError(
            "No cv2.HoughLinesP calls intercepted. "
            "Check the detector's Hough invocation."
        )

    result["hough_calls"] = calls["count"]
    result["removed_edge_pixels"] = (
        masked_edge_pixels["count"]
    )

    return result


# ==================================================
# 6. VISUALIZATION
# ==================================================

def draw_fit(image, fit, color):

    if fit is None:
        return

    xb, yb, xt, yt = map(float, fit)

    if abs(yb - yt) < 1e-9:
        return

    slope = (xb - xt) / (yb - yt)

    def point(y):
        x = xt + slope * (y - yt)
        return int(round(x)), int(round(y))

    height = image.shape[0]

    cv2.line(
        image,
        point(0.35 * height),
        point(0.95 * height),
        color,
        3,
    )


def draw_vp(image, vp, color, label):

    if vp is None:
        return

    x, y = vp

    if not np.isfinite([x, y]).all():
        return

    height, width = image.shape[:2]

    if not (
        0 <= x < width and 0 <= y < height
    ):
        return

    center = (
        int(round(x)),
        int(round(y)),
    )

    cv2.circle(
        image,
        center,
        8,
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )


def diagnostic_image(
    frame,
    result,
    reference,
    method,
    frame_number,
):

    image = frame.copy()

    draw_fit(
        image,
        result["left_fit"],
        (0, 255, 0),
    )

    draw_fit(
        image,
        result["right_fit"],
        (255, 0, 0),
    )

    draw_vp(
        image,
        result["vp"],
        (0, 165, 255),
        "AUTO",
    )

    draw_vp(
        image,
        reference,
        (255, 0, 255),
        "MANUAL",
    )

    cv2.rectangle(
        image,
        (10, 10),
        (850, 105),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        f"Frame {frame_number} - {method}",
        (25, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"VP: {result['vp_reason']}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    return cv2.resize(
        image,
        (640, 360),
    )


# ==================================================
# 7. PROCESS REFERENCE FRAMES
# ==================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

rows = []

print("\n--- EDGE-MASKED VP EXPERIMENT ---")
print(f"Reference frames: {len(MANUAL_REFERENCES)}")

try:

    for frame_number, reference in (
        MANUAL_REFERENCES.items()
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1,
        )

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(
                f"Cannot read frame {frame_number}"
            )

        observations = tracks_by_frame.get(
            frame_number,
            tracks.iloc[0:0],
        )

        object_mask = make_object_mask(
            frame,
            observations,
        )

        # A. Original detector
        original = evaluate_lane_geometry(frame)

        # B. RGB inpainting
        if np.any(object_mask):
            inpainted_frame = cv2.inpaint(
                frame,
                object_mask,
                INPAINT_RADIUS,
                cv2.INPAINT_TELEA,
            )
        else:
            inpainted_frame = frame.copy()

        inpainted = evaluate_lane_geometry(
            inpainted_frame
        )

        # C. Canny/Hough edge masking
        edge_masked = evaluate_edge_masked(
            frame,
            object_mask,
        )

        methods = {
            "ORIGINAL": original,
            "INPAINTED": inpainted,
            "EDGE_MASKED": edge_masked,
        }

        diagnostics = []

        for name, result in methods.items():

            vp = result["vp"]

            if result["vp_valid"] and vp is not None:
                error_x = vp[0] - reference[0]
                error_y = vp[1] - reference[1]
            else:
                error_x = np.nan
                error_y = np.nan

            rows.append({
                "frame": frame_number,
                "method": name,
                "vp_valid": result["vp_valid"],
                "vp_reason": result["vp_reason"],
                "lane_valid": result["lane_valid"],
                "vp_x": vp[0] if vp else np.nan,
                "vp_y": vp[1] if vp else np.nan,
                "manual_x": reference[0],
                "manual_y": reference[1],
                "abs_error_x": abs(error_x),
                "abs_error_y": abs(error_y),
                "tracked_objects": len(observations),
                "mask_fraction": (
                    np.count_nonzero(object_mask)
                    / object_mask.size
                ),
                "removed_edge_pixels": result.get(
                    "removed_edge_pixels", np.nan
                ),
            })

            displayed_frame = (
                inpainted_frame
                if name == "INPAINTED"
                else frame
            )

            diagnostics.append(
                diagnostic_image(
                    displayed_frame,
                    result,
                    reference,
                    name,
                    frame_number,
                )
            )

        combined = np.hstack(diagnostics)

        output_image = (
            OUTPUT_DIR
            / f"compare_{frame_number}.jpg"
        )

        cv2.imwrite(
            str(output_image),
            combined,
        )

        print(f"\nFrame {frame_number}")

        for name, result in methods.items():

            vp = result["vp"]

            if result["vp_valid"] and vp is not None:
                error = abs(vp[1] - reference[1])
                error_text = f"{error:.2f} px"
            else:
                error_text = "N/A"

            print(
                f"  {name:12s} "
                f"valid={result['vp_valid']} "
                f"error_y={error_text}"
            )

        print(
            "  Removed edge pixels:",
            edge_masked["removed_edge_pixels"],
        )

finally:
    cap.release()


# ==================================================
# 8. SAVE AND SUMMARIZE
# ==================================================

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\n--- SUMMARY ---")

for method in [
    "ORIGINAL",
    "INPAINTED",
    "EDGE_MASKED",
]:

    subset = df[df["method"] == method]
    valid = subset[subset["vp_valid"]]

    print(f"\n{method}")
    print(f"Valid: {len(valid)}/{len(subset)}")

    if not valid.empty:
        print(
            "Median Y error:",
            round(valid["abs_error_y"].median(), 2),
        )
        print(
            "Mean Y error:",
            round(valid["abs_error_y"].mean(), 2),
        )
        print(
            "Max Y error:",
            round(valid["abs_error_y"].max(), 2),
        )


# Compare all three only where all succeeded.
wide = df.pivot(
    index="frame",
    columns="method",
    values="abs_error_y",
)

matched = wide.dropna(
    subset=[
        "ORIGINAL",
        "INPAINTED",
        "EDGE_MASKED",
    ]
)

print("\n--- THREE-WAY MATCHED COMPARISON ---")
print(f"Matched frames: {len(matched)}")

if not matched.empty:
    print(matched.round(2).to_string())

    print("\nMedian errors on matched frames:")

    print(
        matched.median().round(2).to_string()
    )

print("\n--- OUTPUT ---")
print(OUTPUT_CSV)
print(OUTPUT_DIR)
print("\nExperiment complete!")
