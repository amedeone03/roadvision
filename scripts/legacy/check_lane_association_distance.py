
import cv2
import pandas as pd

from src.lanes.lane_detector import detect_lane
from src.lanes.temporal_filter import TemporalLaneFilter
from src.lanes.lane_smoother import LaneSmoother

from src.association.vehicle_lane import (
    build_lane_polygon,
)


VIDEO_PATH = "data/videos/test_drive.mp4"
CSV_PATH = "runs/integrated_restored.csv"

TARGET_FRAME = 2


def main():

    # ------------------------------------------
    # 1. Load tracked vehicle observations
    # ------------------------------------------

    df = pd.read_csv(CSV_PATH)

    objects = df[df["frame"] == TARGET_FRAME]

    if objects.empty:
        raise RuntimeError(
            f"No tracked objects in frame {TARGET_FRAME}"
        )

    # ------------------------------------------
    # 2. Initialize lane pipeline
    # ------------------------------------------

    lane_filter = TemporalLaneFilter(
        max_jump=100,
        max_reference_age=5,
        max_candidate_gap=2,
    )

    lane_smoother = LaneSmoother(alpha=0.3)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    polygon = None

    try:
        # Process every frame up to the target,
        # preserving temporal state.
        for frame_number in range(1, TARGET_FRAME + 1):

            success, frame = cap.read()

            if not success:
                raise RuntimeError(
                    f"Cannot read frame {frame_number}"
                )

            _, lane_info = detect_lane(frame)

            accepted, reason = lane_filter.update(
                frame_number,
                lane_info,
            )

            left_fit, right_fit = lane_smoother.update(
                lane_info,
                accepted,
                reason,
            )

            if frame_number == TARGET_FRAME:

                print(
                    f"Frame {frame_number}: "
                    f"accepted={accepted}, reason={reason}"
                )

                if accepted:
                    polygon = build_lane_polygon(
                        left_fit,
                        right_fit,
                    )

    finally:
        cap.release()

    if polygon is None:
        print("No accepted polygon in target frame")
        return

    # ------------------------------------------
    # 3. Calculate signed distances
    # ------------------------------------------

    print("\n--- VEHICLE-TO-LANE DISTANCES ---")

    for _, obj in objects.iterrows():

        contact_point = (
            float(obj["contact_x"]),
            float(obj["contact_y"]),
        )

        distance = cv2.pointPolygonTest(
            polygon,
            contact_point,
            True,
        )

        print(
            f"Track {int(obj['track_id'])} "
            f"({obj['class_name']}): "
            f"signed distance = {distance:.2f} px | "
            f"status = {obj['lane_status']}"
        )

    print("\nInterpretation:")
    print("  Positive: contact point inside polygon")
    print("  Negative: contact point outside polygon")
    print("  Near zero: contact point close to boundary")


if __name__ == "__main__":
    main()
