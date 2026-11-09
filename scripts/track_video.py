
from pathlib import Path
import csv
import time

import cv2
import torch
from ultralytics import YOLO


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

# Input video
VIDEO_PATH = Path("data/videos/test_drive.mp4")

# Selected model and tracker
MODEL_PATH = "yolo11n.pt"
TRACKER = "configs/bytetrack_threshold.yaml"

# Detection settings
CONFIDENCE = 0.25

# COCO classes:
# 0 = person, 1 = bicycle, 2 = car,
# 3 = motorcycle, 5 = bus, 7 = truck
ROAD_USER_CLASSES = [0, 1, 2, 3, 5, 7]

# Output files
OUTPUT_PATH = Path("runs/tracking_selected.mp4")
CSV_PATH = Path("runs/tracking_selected.csv")

# Hardware acceleration
DEVICE = "mps"


# --------------------------------------------------
# 2. Load YOLO model
# --------------------------------------------------

model = YOLO(MODEL_PATH)

print(f"Model: {MODEL_PATH}")
print(f"Tracker: {TRACKER}")
print(f"Device: {DEVICE}")


# --------------------------------------------------
# 3. Open input video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video metadata")

print(f"Video: {width}x{height} at {fps:.1f} FPS")


# --------------------------------------------------
# 4. Prepare output files
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Cannot create output video: {OUTPUT_PATH}"
    )


# --------------------------------------------------
# 5. Initialize tracking statistics
# --------------------------------------------------

frame_count = 0
total_time = 0.0
total_detections = 0

unique_ids = set()


# --------------------------------------------------
# 6. Process video frame by frame
# --------------------------------------------------

try:
    with open(CSV_PATH, "w", newline="") as csv_file:

        csv_writer = csv.writer(csv_file)

        # CSV header
        csv_writer.writerow([
            "frame",
            "timestamp",
            "track_id",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ])

        while True:

            # Read next frame
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # Timestamp in seconds
            timestamp = (frame_count - 1) / fps

            # Start processing timer
            start = time.perf_counter()

            # ----------------------------------
            # YOLO detection + ByteTrack
            # ----------------------------------

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER,
                conf=CONFIDENCE,
                classes=ROAD_USER_CLASSES,
                device=DEVICE,
                verbose=False,
            )

            result = results[0]

            # ----------------------------------
            # Extract tracked objects
            # ----------------------------------

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                boxes = result.boxes

                track_ids = (
                    boxes.id.int().cpu().tolist()
                )

                class_ids = (
                    boxes.cls.int().cpu().tolist()
                )

                confidences = (
                    boxes.conf.cpu().tolist()
                )

                coordinates = (
                    boxes.xyxy.cpu().tolist()
                )

                # Save each tracked object to CSV
                for track_id, class_id, conf, box in zip(
                    track_ids,
                    class_ids,
                    confidences,
                    coordinates,
                ):

                    x1, y1, x2, y2 = box

                    class_name = model.names[class_id]

                    csv_writer.writerow([
                        frame_count,
                        round(timestamp, 3),
                        track_id,
                        class_name,
                        round(conf, 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ])

                    unique_ids.add(track_id)
                    total_detections += 1

            # ----------------------------------
            # Annotate frame
            # ----------------------------------

            annotated = result.plot()

            # Ensure GPU processing is finished
            # before stopping the timer
            if DEVICE == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start
            total_time += elapsed

            # Write annotated frame
            writer.write(annotated)

            # Print progress every 30 frames
            if frame_count % 30 == 0:

                print(
                    f"Processed {frame_count} frames | "
                    f"Unique IDs: {len(unique_ids)}"
                )

finally:

    # Always release video resources
    cap.release()
    writer.release()


# --------------------------------------------------
# 7. Final performance summary
# --------------------------------------------------

if frame_count > 0:

    avg_time = total_time / frame_count

    processing_fps = 1 / avg_time

    print("\n--- TRACKING COMPLETE ---")

    print(f"Frames processed: {frame_count}")

    print(
        f"Total tracked detections: "
        f"{total_detections}"
    )

    print(
        f"Unique track IDs: "
        f"{len(unique_ids)}"
    )

    print(
        f"Average processing time: "
        f"{avg_time * 1000:.1f} ms"
    )

    print(
        f"Processing FPS: "
        f"{processing_fps:.1f}"
    )

    print(f"Video saved to: {OUTPUT_PATH}")
    print(f"CSV saved to: {CSV_PATH}")

else:

    print("No frames were processed.")
