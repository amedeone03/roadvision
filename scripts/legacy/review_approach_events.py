
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO = Path("data/videos/test_drive.mp4")
OBSERVATIONS = Path("runs/phase6/lane_evidence_results.csv")
EVENTS = Path("runs/phase6/lane_approach_events.csv")

OUTPUT_DIR = Path("runs/phase6/event_review")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_WIDTH = 640
PANEL_HEIGHT = 360


def read_frame(cap, frame_number):
    # CSV frame numbers are 1-based.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(
            f"Unable to read frame {frame_number}"
        )

    return frame


def draw_observation(frame, row, title):
    image = frame.copy()
    height, width = image.shape[:2]

    coords = np.array(
        [row.x1, row.y1, row.x2, row.y2],
        dtype=float,
    )

    if np.isfinite(coords).all():
        x1, y1, x2, y2 = np.round(coords).astype(int)

        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height - 1))

        if x2 > x1 and y2 > y1:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                4,
            )

    tau = (
        f"{row.tau_proxy_s:.2f}s"
        if pd.notna(row.tau_proxy_s)
        else "N/A"
    )

    cv2.rectangle(
        image,
        (12, 12),
        (1050, 130),
        (15, 15, 15),
        -1,
    )

    cv2.putText(
        image,
        title,
        (28, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"ID {int(row.track_id)} | "
        f"{row.lane_status} | tau proxy {tau}",
        (28, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    return cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
    )


def main():
    events = pd.read_csv(EVENTS)
    observations = pd.read_csv(OBSERVATIONS)

    required = {
        "frame", "track_id", "tau_proxy_s",
        "lane_status", "x1", "y1", "x2", "y2",
    }

    missing = required - set(observations.columns)

    if missing:
        raise ValueError(
            f"Missing observation columns: {missing}"
        )

    if observations.duplicated(["frame", "track_id"]).any():
        raise ValueError(
            "Duplicate frame/track observations"
        )

    lookup = observations.set_index(
        ["track_id", "frame"],
        drop=False,
    )

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")

    try:
        for event_index, event in events.iterrows():
            track_id = int(event.track_id)
            start = int(event.start_frame)
            end = int(event.end_frame)

            segment = observations[
                (observations["track_id"] == track_id)
                & observations["frame"].between(start, end)
            ]

            if segment.empty:
                raise RuntimeError(
                    f"Empty episode: track {track_id}, {start}-{end}"
                )

            peak_frame = int(
                segment.loc[
                    segment["tau_proxy_s"].idxmin(),
                    "frame",
                ]
            )

            selections = [
                ("START", start),
                ("PEAK", peak_frame),
                ("END", end),
            ]

            panels = []

            for label, frame_number in selections:
                frame = read_frame(cap, frame_number)
                row = lookup.loc[(track_id, frame_number)]

                panel = draw_observation(
                    frame,
                    row,
                    f"{label} | Frame {frame_number}",
                )

                panels.append(panel)

            combined = np.hstack(panels)

            output = OUTPUT_DIR / (
                f"event_{event_index + 1:02d}_"
                f"track_{track_id}_{start}_{end}.jpg"
            )

            if not cv2.imwrite(str(output), combined):
                raise RuntimeError(f"Could not write {output}")

            print(
                f"Event {event_index + 1}: "
                f"Track {track_id}, frames {start}-{end}, "
                f"peak at {peak_frame}"
            )

    finally:
        cap.release()

    print(f"\nSaved {len(events)} event images to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
