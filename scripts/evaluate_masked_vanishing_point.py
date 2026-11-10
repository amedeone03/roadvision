
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
TRACKING_CSV = Path("runs/integrated_final.csv")

OUTPUT_DIR = Path(
    "runs/motion/masked_vanishing_point"
)

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

# Extend masks slightly beyond YOLO bounding boxes.
BOX_PADDING = 0.03

# Inpainting parameter in pixels.
INPAINT_RADIUS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD TRACKING DATA
# ==================================================

if not TRACKING_CSV.exists():
    raise FileNotFoundError(TRACKING_CSV)

tracks = pd.read_csv(TRACKING_CSV)

required = {
    "frame", "track_id",
    "x1", "y1", "x2", "y2",
}

missing = required - set(tracks.columns)

if missing:
    raise ValueError(
        f"Missing tracking columns: {sorted(missing)}"
    )

# Organize observations by frame for fast lookup.
tracks_by_frame = {
    int(frame): group
    for frame, group in tracks.groupby("frame")
}


# ==================================================
# 3. VEHICLE MASKING
# ==================================================

def make_object_mask(frame, observations):
    """
    Generate a binary mask from tracked-object
    bounding boxes.

    White (255): region to suppress.
    Black (0): preserve original pixels.
    """

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


def suppress_objects(frame, mask):
    """
    Replace masked pixels using OpenCV inpainting.

    Inpainting is an experimental preprocessing
    method; it does not reconstruct true road
    markings hidden behind vehicles.
    """

    if not np.any(mask):
        return frame.copy()

    return cv2.inpaint(
        frame,
        mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA,
    )


# ==================================================
# 4. AUTOMATIC VP ESTIMATION
# ==================================================

def evaluate_lane_geometry(frame):
    """
    Use the existing Phase 3 detector.

    Returns automatic VP and diagnostic metadata.
    """

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
        "vp": vp,
        "valid": valid,
        "reason": reason,
        "lane_valid": lane_valid,
        "left_fit": left_fit,
        "right_fit": right_fit,
    }


# ==================================================
# 5. VISUALIZATION
# ==================================================

def draw_fit(image, fit, color):

    if fit is None:
        return

    xb, yb, xt, yt = map(float, fit)

    if abs(yb - yt) < 1e-9:
        return

    slope = (xb - xt) / (yb - yt)

    def point_at_y(y):
        x = xt + slope * (y - yt)
        return int(round(x)), int(round(y))

    height = image.shape[0]

    cv2.line(
        image,
        point_at_y(0.35 * height),
        point_at_y(0.95 * height),
        color,
        3,
    )


def draw_point(image, vp, color, label):

    if vp is None:
        return

    x, y = vp

    if not np.isfinite([x, y]).all():
        return

    height, width = image.shape[:2]

    if not (0 <= x < width and 0 <= y < height):
        return

    center = int(round(x)), int(round(y))

    cv2.circle(
        image, center, 9, color, -1
    )

    cv2.putText(
        image,
        label,
        (center[0] + 12, center[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def make_diagnostic_image(
    frame,
    result,
    manual_vp,
    title,
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

    draw_point(
        image,
        result["vp"],
        (0, 165, 255),
        "AUTO",
    )

    draw_point(
        image,
        manual_vp,
        (255, 0, 255),
        "MANUAL",
    )

    cv2.rectangle(
        image,
        (10, 10),
        (900, 100),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        title,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"VP: {result['reason']}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    return image


# ==================================================
# 6. PROCESS REFERENCE FRAMES
# ==================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )

rows = []

print("\n--- OCCLUSION-AWARE VP EXPERIMENT ---")
print(f"Video: {VIDEO_PATH}")
print(f"Reference frames: {len(MANUAL_REFERENCES)}")

try:

    for frame_number, manual_vp in (
        MANUAL_REFERENCES.items()
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1,
        )

        ok, frame = cap.read()

        if not ok:
            print(
                f"Frame {frame_number}: read failed"
            )
            continue

        observations = tracks_by_frame.get(
            frame_number,
            pd.DataFrame(
                columns=list(required)
            ),
        )

        # -------------------------------
        # ORIGINAL METHOD
        # -------------------------------

        original = evaluate_lane_geometry(frame)

        # -------------------------------
        # MASKED METHOD
        # -------------------------------

        mask = make_object_mask(
            frame,
            observations,
        )

        masked_frame = suppress_objects(
            frame,
            mask,
        )

        masked = evaluate_lane_geometry(
            masked_frame
        )

        # -------------------------------
        # ERROR AGAINST MANUAL REFERENCE
        # -------------------------------

        for method, result in [
            ("ORIGINAL", original),
            ("MASKED", masked),
        ]:

            vp = result["vp"]

            if result["valid"] and vp is not None:

                error_x = vp[0] - manual_vp[0]
                error_y = vp[1] - manual_vp[1]

            else:
                error_x = np.nan
                error_y = np.nan

            rows.append({
                "frame": frame_number,
                "method": method,

                "tracked_objects": len(
                    observations
                ),

                "masked_pixel_fraction": (
                    np.count_nonzero(mask)
                    / mask.size
                ),

                "lane_valid": result["lane_valid"],

                "vp_valid": result["valid"],
                "vp_reason": result["reason"],

                "vp_x": (
                    vp[0]
                    if vp is not None
                    else np.nan
                ),

                "vp_y": (
                    vp[1]
                    if vp is not None
                    else np.nan
                ),

                "manual_x": manual_vp[0],
                "manual_y": manual_vp[1],

                "error_x": error_x,
                "error_y": error_y,

                "abs_error_x": abs(error_x),
                "abs_error_y": abs(error_y),
            })

        # -------------------------------
        # SAVE IMAGES
        # -------------------------------

        original_image = make_diagnostic_image(
            frame,
            original,
            manual_vp,
            f"Frame {frame_number} - ORIGINAL",
        )

        masked_image = make_diagnostic_image(
            masked_frame,
            masked,
            manual_vp,
            f"Frame {frame_number} - MASKED",
        )

        mask_overlay = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR,
        )

        comparison = np.hstack([
            cv2.resize(
                original_image,
                (960, 540),
            ),
            cv2.resize(
                masked_image,
                (960, 540),
            ),
        ])

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"compare_{frame_number}.jpg"
            ),
            comparison,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"mask_{frame_number}.png"
            ),
            mask_overlay,
        )

        # -------------------------------
        # PRINT PROGRESS
        # -------------------------------

        def format_error(result):
            if not result["valid"]:
                return "N/A"

            return (
                f"{abs(result['vp'][1] - manual_vp[1]):.2f}"
            )

        print(
            f"Frame {frame_number}: "
            f"objects={len(observations)} | "
            f"original_error_y={format_error(original)} | "
            f"masked_error_y={format_error(masked)}"
        )

finally:
    cap.release()


# ==================================================
# 7. QUANTITATIVE COMPARISON
# ==================================================

if not rows:
    raise RuntimeError("No reference frames processed")

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\n--- METHOD COMPARISON ---")

for method in ["ORIGINAL", "MASKED"]:

    subset = df[df["method"] == method]

    valid = subset[subset["vp_valid"]]

    print(f"\n{method}")
    print(f"Valid VP: {len(valid)}/{len(subset)}")

    if not valid.empty:

        print(
            "Median absolute Y error: "
            f"{valid['abs_error_y'].median():.2f} px"
        )

        print(
            "Mean absolute Y error: "
            f"{valid['abs_error_y'].mean():.2f} px"
        )

        print(
            "Maximum absolute Y error: "
            f"{valid['abs_error_y'].max():.2f} px"
        )


# ==================================================
# 8. PAIRED COMPARISON
# ==================================================

wide = df.pivot(
    index="frame",
    columns="method",
    values="abs_error_y",
)

paired = wide.dropna(
    subset=["ORIGINAL", "MASKED"]
)

print("\n--- PAIRED REFERENCE COMPARISON ---")
print(f"Frames valid for both: {len(paired)}")

if not paired.empty:

    improvement = (
        paired["ORIGINAL"] - paired["MASKED"]
    )

    print(
        "Masked improved:",
        int((improvement > 0).sum()),
    )

    print(
        "Masked worsened:",
        int((improvement < 0).sum()),
    )

    print(
        "Equal:",
        int((improvement == 0).sum()),
    )

    print(
        "Median error reduction: "
        f"{improvement.median():+.2f} px"
    )

    print("\nPer-frame errors:")
    print(paired.round(2).to_string())


# ==================================================
# 9. FINISH
# ==================================================

print("\n--- OUTPUT ---")
print(f"CSV: {OUTPUT_CSV}")
print(f"Images: {OUTPUT_DIR}")

print("\nMasked VP experiment complete!")
