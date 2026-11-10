
from pathlib import Path

import cv2


VIDEO_PATH = Path("runs/integrated_restored_output.mp4")
OUTPUT_DIR = Path("runs/association_debug")

# Inclusive frame ranges, matching our 1-based CSV frame numbers.
CLIPS = {
    "track_8_flips": (295, 330),
    "track_125_flips": (995, 1025),
}


def extract_clip(cap, start_frame, end_frame, output_path, fps, width, height):
    """Extract a range of frames from an existing video."""

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create output video: {output_path}"
        )

    # OpenCV uses zero-based frame indices.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    count = 0

    try:
        for frame_number in range(start_frame, end_frame + 1):

            success, frame = cap.read()

            if not success:
                break

            cv2.putText(
                frame,
                f"FRAME {frame_number}",
                (30, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            writer.write(frame)
            count += 1

    finally:
        writer.release()

    print(f"{output_path}: {count} frames")


def main():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Invalid video metadata")

        for name, (start_frame, end_frame) in CLIPS.items():

            output_path = OUTPUT_DIR / f"{name}.mp4"

            extract_clip(
                cap,
                start_frame,
                end_frame,
                output_path,
                fps,
                width,
                height,
            )

    finally:
        cap.release()

    print("\nClip extraction complete!")


if __name__ == "__main__":
    main()
