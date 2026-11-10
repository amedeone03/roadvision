
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.lanes.lane_detector import detect_lane
from src.association.vehicle_lane import (
    build_extended_lane_polygon,
    classify_vehicle_lane,
    get_bottom_center,
)


# ------------------------------------------
# 1. Configuration
# ------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_PATH = Path("runs/integrated_frame.png")

MODEL_PATH = "yolo11n.pt"
DEVICE = "mps"

TARGET_TIME = 5.0

# COCO road-user classes:
# person, bicycle, car, motorcycle, bus, truck
ROAD_USER_CLASSES = [0, 1, 2, 3, 5, 7]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------
# 2. Read the selected video frame
# ------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid input FPS")

target_frame = int(TARGET_TIME * fps)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    target_frame,
)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read target frame")


# ------------------------------------------
# 3. Run lane detection
# ------------------------------------------

annotated, lane_info = detect_lane(frame)

lane_polygon = None

if lane_info["valid"]:
    lane_polygon = build_extended_lane_polygon(
        lane_info["left_fit"],
        lane_info["right_fit"],
        y_top=int(0.55 * frame.shape[0]),
    )


# ------------------------------------------
# 4. Load YOLO
# ------------------------------------------

model = YOLO(MODEL_PATH)

results = model.predict(
    frame,
    conf=0.25,
    classes=ROAD_USER_CLASSES,
    device=DEVICE,
    verbose=False,
)

result = results[0]


# ------------------------------------------
# 5. Associate detections with ego lane
# ------------------------------------------

counts = {
    "IN_LANE": 0,
    "OUT_LANE": 0,
    "UNKNOWN": 0,
}

if result.boxes is not None:

    for box in result.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy()
        )

        bbox = (x1, y1, x2, y2)

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        class_name = model.names[class_id]

        lane_status = classify_vehicle_lane(
            bbox,
            lane_polygon,
        )

        counts[lane_status] += 1

        # ----------------------------------
        # Visualization color
        # ----------------------------------

        if lane_status == "IN_LANE":
            color = (0, 255, 0)

        elif lane_status == "OUT_LANE":
            color = (0, 165, 255)

        else:
            color = (160, 160, 160)

        # ----------------------------------
        # Draw bounding box
        # ----------------------------------

        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )

        label = (
            f"{class_name} {confidence:.2f} "
            f"| {lane_status}"
        )

        cv2.putText(
            annotated,
            label,
            (int(x1), max(int(y1) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

        # ----------------------------------
        # Draw bottom-center contact point
        # ----------------------------------

        cx, cy = get_bottom_center(bbox)

        cv2.circle(
            annotated,
            (int(cx), int(cy)),
            5,
            color,
            -1,
        )


# ------------------------------------------
# 6. Save result
# ------------------------------------------

if not cv2.imwrite(str(OUTPUT_PATH), annotated):
    raise RuntimeError("Could not save output image")

print("\n--- INTEGRATED FRAME RESULTS ---")

print(f"Frame: {target_frame}")
print(f"Timestamp: {TARGET_TIME:.2f} s")
print(f"Lane geometry valid: {lane_info['valid']}")

print(f"IN_LANE: {counts['IN_LANE']}")
print(f"OUT_LANE: {counts['OUT_LANE']}")
print(f"UNKNOWN: {counts['UNKNOWN']}")

print(f"Output: {OUTPUT_PATH}")


print("\n--- ASSOCIATION POLYGON DEBUG ---")
print(f"Left fit: {lane_info['left_fit']}")
print(f"Right fit: {lane_info['right_fit']}")
print(f"Extended polygon valid: {lane_polygon is not None}")
print(f"Extended polygon: {lane_polygon}")