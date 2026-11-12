
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

VIDEO_PATH = Path("data/videos/test_drive.mp4")

INPUT_CSV = Path(
    "runs/phase6/stabilized_evidence_results.csv"
)

OUTPUT_DIR = Path("runs/phase6/final")
OUTPUT_VIDEO = OUTPUT_DIR / "phase6_output.mp4"
OUTPUT_CSV = OUTPUT_DIR / "phase6_results.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANE_COLORS = {
    "IN_LANE": (0, 220, 255),      # Yellow
    "OUT_LANE": (120, 210, 120),   # Green
    "UNKNOWN": (180, 180, 180),    # Gray
}

EVENT_STATES = {
    "NONE",
    "PENDING",
    "CONFIRMED",
    "HELD",
}


# ==================================================
# DRAWING HELPERS
# ==================================================

def draw_text(
    image,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.56,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )
    )

    height, width = image.shape[:2]

    x = int(np.clip(
        x,
        0,
        max(0, width - text_width - 14),
    ))

    y = int(np.clip(
        y,
        text_height + 12,
        max(text_height + 12, height - baseline - 8),
    ))

    cv2.rectangle(
        image,
        (x, y - text_height - 8),
        (x + text_width + 12, y + baseline + 6),
        (18, 18, 18),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 6, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_vehicle(frame, row):
    height, width = frame.shape[:2]

    box = np.array(
        [row.x1, row.y1, row.x2, row.y2],
        dtype=float,
    )

    if not np.isfinite(box).all():
        return

    x1, y1, x2, y2 = (
        np.round(box).astype(int)
    )

    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))

    if x2 <= x1 or y2 <= y1:
        return

    lane = str(row.lane_status)
    motion = str(row.approach_t015)
    event = str(row.event_state)

    color = LANE_COLORS.get(
        lane,
        LANE_COLORS["UNKNOWN"],
    )

    # The box color describes lane classification,
    # not estimated collision risk.
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    draw_text(
        frame,
        f"ID {int(row.track_id)} | {lane}",
        x1,
        y1 - 12,
        color=color,
    )

    # Full motion annotations for ego-lane objects.
    if lane == "IN_LANE":

        draw_text(
            frame,
            f"Scale: {motion}",
            x1,
            y2 + 24,
        )

        if event in {
            "PENDING",
            "CONFIRMED",
            "HELD",
        }:

            draw_text(
                frame,
                f"Evidence: {event}",
                x1,
                y2 + 52,
                color=(
                    (0, 220, 255)
                    if event == "CONFIRMED"
                    else (200, 200, 200)
                ),
            )

        if (
            event == "CONFIRMED"
            and pd.notna(row.tau_proxy_s)
        ):

            draw_text(
                frame,
                (
                    "Inverse expansion: "
                    f"{row.tau_proxy_s:.2f} s"
                ),
                x1,
                y2 + 80,
            )

    # If the current lane is uncertain, do not
    # display HELD as current ego-lane confirmation.
    elif event == "HELD":

        draw_text(
            frame,
            "Previous approach evidence; lane uncertain",
            x1,
            y2 + 24,
            color=(190, 190, 190),
            scale=0.48,
        )


# ==================================================
# MAIN
# ==================================================

def main():

    df = pd.read_csv(INPUT_CSV)

    required = {
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "lane_status",
        "approach_t015",
        "tau_proxy_s",
        "event_state",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.duplicated(
        ["frame", "track_id"]
    ).any():
        raise ValueError(
            "Duplicate frame/track observations"
        )

    if not df["event_state"].isin(
        EVENT_STATES
    ).all():
        raise ValueError(
            "Unexpected event state"
        )

    # Confirmed means current positive evidence.
    confirmed = df[
        df["event_state"] == "CONFIRMED"
    ]

    assert (
        confirmed["lane_status"] == "IN_LANE"
    ).all()

    assert (
        confirmed["approach_t015"] == "APPROACHING"
    ).all()

    assert confirmed["tau_proxy_s"].notna().all()

    # Export the complete analytical results.
    df = df.sort_values(
        ["frame", "track_id"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    grouped = {
        int(frame_number): group
        for frame_number, group
        in df.groupby("frame")
    }

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {VIDEO_PATH}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

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

    frame_number = 0
    annotated_observations = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            observations = grouped.get(
                frame_number
            )

            if observations is not None:

                for row in observations.itertuples(
                    index=False
                ):
                    draw_vehicle(frame, row)
                    annotated_observations += 1

            # Global experimental disclaimer.
            cv2.rectangle(
                frame,
                (12, 12),
                (950, 85),
                (18, 18, 18),
                -1,
            )

            cv2.putText(
                frame,
                "RoadVision | Phase 6: Approach Evidence",
                (26, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Experimental image-space indicators - not collision warnings",
                (26, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (180, 200, 255),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)

            if frame_number % 200 == 0:
                print(
                    f"Processed {frame_number} frames"
                )

    finally:
        cap.release()
        writer.release()

    print("\n--- PHASE 6 INTEGRATION ---")

    print("Frames:", frame_number)
    print(
        "Annotated observations:",
        annotated_observations,
    )

    print(
        "Unique tracks:",
        df["track_id"].nunique(),
    )

    print("\nEvent states:")
    print(
        df["event_state"]
        .value_counts()
        .to_string()
    )

    print("\nOutputs:")
    print(OUTPUT_VIDEO)
    print(OUTPUT_CSV)

    print("\nPhase 6 output generated!")


if __name__ == "__main__":
    main()
