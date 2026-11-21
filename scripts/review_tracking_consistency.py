
from pathlib import Path

import cv2
import pandas as pd


VIDEO = Path("data/videos/test_drive.mp4")
TRACKS = Path("runs/phase6/final/phase6_results.csv")
OUTPUT = Path("runs/phase7/tracking/visual_review")

# Each case covers a potential interruption or ID change.
CASES = {
    "tracks_9_to_10": [63, 66, 70, 73, 76, 80],
    "tracks_1_to_12": [108, 111, 112, 113, 116, 120],
    "tracks_62_to_66": [555, 558, 560, 562, 565, 570],
    "track_66_event": [660, 665, 667, 668, 670, 675],
}

TARGET_TRACKS = {
    "tracks_9_to_10": {9, 10},
    "tracks_1_to_12": {1, 12},
    "tracks_62_to_66": {62, 66},
    "track_66_event": {66},
}


def read_frame(cap, frame_number):
    # RoadVision uses 1-based frame numbers.
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number - 1,
    )

    success, frame = cap.read()

    if not success:
        raise RuntimeError(
            f"Cannot read frame {frame_number}"
        )

    return frame


def annotate(frame, frame_number, tracks, targets):
    result = frame.copy()

    observations = tracks[
        tracks["frame"] == frame_number
    ]

    for row in observations.itertuples(index=False):
        track_id = int(row.track_id)

        if track_id not in targets:
            continue

        x1, y1, x2, y2 = [
            int(round(value))
            for value in [
                row.x1, row.y1, row.x2, row.y2
            ]
        ]

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            result,
            f"ID {track_id} | {row.class_name}",
            (x1, max(35, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        result,
        f"Frame {frame_number}",
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(TRACKS)

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")

    try:
        for case_name, frames in CASES.items():

            case_dir = OUTPUT / case_name
            case_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            targets = TARGET_TRACKS[case_name]

            for frame_number in frames:
                frame = read_frame(cap, frame_number)

                annotated = annotate(
                    frame,
                    frame_number,
                    tracks,
                    targets,
                )

                # Smaller files for visual review.
                preview = cv2.resize(
                    annotated,
                    (960, 540),
                )

                output_path = (
                    case_dir
                    / f"frame_{frame_number:04d}.jpg"
                )

                if not cv2.imwrite(
                    str(output_path),
                    preview,
                ):
                    raise RuntimeError(
                        f"Cannot save {output_path}"
                    )

            print(
                f"{case_name}: {len(frames)} images"
            )

    finally:
        cap.release()

    print("\nVisual review saved to:", OUTPUT)


if __name__ == "__main__":
    main()
