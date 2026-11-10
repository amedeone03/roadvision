
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO_PATH = Path("data/videos/test_drive.mp4")

INPUT_CSV = Path(
    "runs/motion/motion_reliability_results.csv"
)

OUTPUT_DIR = Path("runs/phase5")
OUTPUT_VIDEO = OUTPUT_DIR / "phase5_output.mp4"
OUTPUT_CSV = OUTPUT_DIR / "phase5_results.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_label(image, text, x, y, color):
    """Draw a readable label with a background."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    height, width = image.shape[:2]

    x = int(np.clip(x, 0, max(0, width - text_w - 12)))
    y = int(np.clip(y, text_h + 12, height - baseline - 5))

    cv2.rectangle(
        image,
        (x, y - text_h - 8),
        (x + text_w + 10, y + baseline + 5),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 5, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def lane_color(lane_status):
    if lane_status == "IN_LANE":
        return (0, 220, 255)

    if lane_status == "OUT_LANE":
        return (120, 210, 120)

    return (180, 180, 180)


def annotate_vehicle(frame, row):

    h, w = frame.shape[:2]

    coordinates = [
        row.x1, row.y1, row.x2, row.y2
    ]

    if not np.isfinite(coordinates).all():
        return

    x1, y1, x2, y2 = (
        int(round(v)) for v in coordinates
    )

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return

    lane_status = str(row.lane_status)
    quality = str(row.motion_quality)

    color = lane_color(lane_status)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    draw_label(
        frame,
        f"ID {int(row.track_id)} | {lane_status}",
        x1,
        max(y1 - 8, 30),
        color,
    )

    # Only display qualified experimental
    # distance and closing-speed estimates.
    if quality == "LOW_SENSITIVITY":

        z = row.distance_dynamic_m
        v = row.closing_speed_qualified_m_s

        if pd.notna(z) and pd.notna(v):

            label = (
                f"EXP Z~{z:.1f}m "
                f"closing~{v:+.1f}m/s"
            )

            draw_label(
                frame,
                label,
                x1,
                min(y2 + 24, h - 10),
                (255, 255, 255),
            )

    elif quality == "UNCERTAIN":

        draw_label(
            frame,
            "Range rate: uncertain",
            x1,
            min(y2 + 24, h - 10),
            (150, 190, 255),
        )


def main():

    df = pd.read_csv(INPUT_CSV)

    required = {
        "frame",
        "track_id",
        "x1", "y1", "x2", "y2",
        "lane_status",
        "motion_quality",
        "distance_dynamic_m",
        "closing_speed_qualified_m_s",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.duplicated(["frame", "track_id"]).any():
        raise ValueError(
            "Duplicate frame/track observations"
        )

    df = df.sort_values(
        ["frame", "track_id"]
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)

    grouped = {
        int(frame_number): observations
        for frame_number, observations
        in df.groupby("frame")
    }

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Cannot create output video"
        )

    processed = 0
    annotated = 0

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            processed += 1

            observations = grouped.get(
                processed
            )

            if observations is not None:

                for row in observations.itertuples(
                    index=False
                ):

                    annotate_vehicle(frame, row)
                    annotated += 1

            # Explicit experimental disclaimer.
            cv2.rectangle(
                frame,
                (12, 12),
                (790, 72),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                frame,
                "RoadVision | Phase 5 Experimental",
                (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Metric estimates are uncalibrated - not safety validated",
                (25, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (180, 200, 255),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)

            if processed % 200 == 0:
                print(
                    f"Processed {processed} frames"
                )

    finally:
        cap.release()
        writer.release()

    print("\n--- PHASE 5 INTEGRATION ---")
    print("Frames processed:", processed)
    print("Annotated observations:", annotated)
    print("Unique tracks:", df["track_id"].nunique())

    print("\nMotion quality:")
    print(
        df["motion_quality"]
        .value_counts()
        .to_string()
    )

    print("\nOutputs:")
    print(OUTPUT_VIDEO)
    print(OUTPUT_CSV)

    print("\nPhase 5 output generated!")


if __name__ == "__main__":
    main()
