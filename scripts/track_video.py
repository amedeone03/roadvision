
import argparse
import csv
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# --------------------------------------------------
# 1. Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="RoadVision multi-object tracking"
)

parser.add_argument(
    "--video",
    type=Path,
    default=Path("data/videos/test_drive.mp4"),
)

parser.add_argument(
    "--model",
    type=str,
    default="yolo11n.pt",
)

parser.add_argument(
    "--tracker",
    type=str,
    default="configs/bytetrack_threshold.yaml",
)

parser.add_argument(
    "--output-name",
    type=str,
    default="selected",
)

parser.add_argument(
    "--conf",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--device",
    type=str,
    default="mps",
)

args = parser.parse_args()

if not 0 <= args.conf <= 1:
    parser.error("--conf must be between 0 and 1")

if not args.output_name.replace("-", "").replace("_", "").isalnum():
    parser.error(
        "--output-name must contain only letters, "
        "numbers, hyphens, or underscores"
    )


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

VIDEO_PATH = args.video
MODEL_PATH = args.model
TRACKER = args.tracker
CONFIDENCE = args.conf
DEVICE = args.device

ROAD_USER_CLASSES = [0, 1, 2, 3, 5, 7]

OUTPUT_DIR = Path("runs")
OUTPUT_PATH = OUTPUT_DIR / f"tracking_{args.output_name}.mp4"
CSV_PATH = OUTPUT_DIR / f"tracking_{args.output_name}.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n--- ROADVISION CONFIGURATION ---")
print(f"Model: {MODEL_PATH}")
print(f"Tracker: {TRACKER}")
print(f"Confidence: {CONFIDENCE}")
print(f"Device: {DEVICE}")
print(f"Video: {VIDEO_PATH}")


# --------------------------------------------------
# 3. Initialize model and video
# --------------------------------------------------

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video metadata")

print(f"Resolution: {width}x{height}")
print(f"Input FPS: {fps:.1f}")


# --------------------------------------------------
# 4. Prepare output video
# --------------------------------------------------

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video")


# --------------------------------------------------
# 5. Tracking statistics
# --------------------------------------------------

frame_count = 0
total_processing_time = 0.0
total_detections = 0
unique_ids = set()

# Measure full loop runtime separately
start_total = time.perf_counter()


# --------------------------------------------------
# 6. Video processing
# --------------------------------------------------

try:
    with CSV_PATH.open("w", newline="") as csv_file:

        csv_writer = csv.writer(csv_file)

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
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1
            timestamp = (frame_count - 1) / fps

            start = time.perf_counter()

            # Detection + tracking
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

            # Save tracked objects
            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes

                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                coordinates = boxes.xyxy.cpu().tolist()

                for track_id, class_id, conf, box in zip(
                    track_ids,
                    class_ids,
                    confidences,
                    coordinates,
                ):
                    x1, y1, x2, y2 = box

                    csv_writer.writerow([
                        frame_count,
                        round(timestamp, 3),
                        track_id,
                        model.names[class_id],
                        round(conf, 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ])

                    unique_ids.add(track_id)
                    total_detections += 1

            # Draw boxes, labels, and IDs
            annotated = result.plot()

            # Synchronize GPU for timing
            if DEVICE == "mps":
                torch.mps.synchronize()
            elif DEVICE.startswith("cuda"):
                torch.cuda.synchronize()

            processing_time = time.perf_counter() - start
            total_processing_time += processing_time

            writer.write(annotated)

            if frame_count % 30 == 0:
                print(
                    f"Frames: {frame_count} | "
                    f"Unique IDs: {len(unique_ids)}"
                )

finally:
    cap.release()
    writer.release()

total_runtime = time.perf_counter() - start_total


# --------------------------------------------------
# 7. Final results
# --------------------------------------------------

print("\n--- TRACKING RESULTS ---")

print(f"Tracker: {TRACKER}")
print(f"Frames processed: {frame_count}")
print(f"Tracked observations: {total_detections}")
print(f"Unique IDs: {len(unique_ids)}")

if frame_count > 0:
    avg_time = total_processing_time / frame_count

    print(
        f"Mean tracking + annotation time: "
        f"{avg_time * 1000:.2f} ms/frame"
    )

    print(
        f"Processing FPS: "
        f"{1 / avg_time:.2f}"
    )

    print(
        f"End-to-end throughput: "
        f"{frame_count / total_runtime:.2f} FPS"
    )

print(f"Video saved to: {OUTPUT_PATH}")
print(f"CSV saved to: {CSV_PATH}")
