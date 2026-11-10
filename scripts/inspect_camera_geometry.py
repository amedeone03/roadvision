
from pathlib import Path

import cv2
import pandas as pd

from src.motion.distance_estimator import CameraModel


VIDEO_PATH = Path("data/videos/test_drive.mp4")
CSV_PATH = Path("runs/motion/distance_scenarios.csv")
OUTPUT_DIR = Path("runs/motion/camera_inspection")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERAS = {
    "pitch_2": CameraModel(1.5, 1000, 540, 2),
    "pitch_3": CameraModel(1.5, 1000, 540, 3),
    "pitch_4": CameraModel(1.5, 1000, 540, 4),
}

COLORS = {
    "pitch_2": (255, 100, 0),
    "pitch_3": (0, 200, 0),
    "pitch_4": (0, 0, 255),
}

# Track 2: reference car ahead.
# Track 127: calibration-sensitive car.
EXAMPLES = [
    (150, 2),
    (1023, 127),
]

df = pd.read_csv(CSV_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

try:
    for frame_number, track_id in EXAMPLES:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1,
        )

        ok, frame = cap.read()

        if not ok:
            print(f"Could not read frame {frame_number}")
            continue

        matching = df[
            (df["frame"] == frame_number)
            & (df["track_id"] == track_id)
        ]

        if matching.empty:
            print(
                f"Track {track_id} absent at "
                f"frame {frame_number}"
            )
            continue

        row = matching.iloc[0]

        # Draw tracked-object bounding box.
        x1, y1, x2, y2 = [
            int(round(row[key]))
            for key in ["x1", "y1", "x2", "y2"]
        ]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3,
        )

        # Draw approximate road-contact point.
        contact = (
            int(round(row["contact_x"])),
            int(round(row["contact_y"])),
        )

        cv2.circle(
            frame,
            contact,
            9,
            (0, 255, 255),
            -1,
        )

        # Draw candidate horizons.
        for name, camera in CAMERAS.items():

            y = int(round(camera.horizon_y))

            cv2.line(
                frame,
                (0, y),
                (frame.shape[1] - 1, y),
                COLORS[name],
                2,
            )

            cv2.putText(
                frame,
                f"{name}: y={y}",
                (30, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                COLORS[name],
                2,
            )

        filename = (
            OUTPUT_DIR
            / f"frame_{frame_number}_track_{track_id}.jpg"
        )

        cv2.imwrite(str(filename), frame)

        print(f"Saved {filename}")

finally:
    cap.release()

print("\nCamera inspection frames created.")
