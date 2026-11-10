
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO = Path("runs/phase5/phase5_output.mp4")
CSV = Path("runs/phase5/phase5_results.csv")

EXPECTED_FRAMES = 1728
EXPECTED_OBSERVATIONS = 2656
EXPECTED_TRACKS = 68

ALLOWED_QUALITY = {
    "UNAVAILABLE",
    "UNCERTAIN",
    "LOW_SENSITIVITY",
}


def main():
    assert VIDEO.exists(), f"Missing video: {VIDEO}"
    assert CSV.exists(), f"Missing CSV: {CSV}"

    df = pd.read_csv(CSV)

    assert len(df) == EXPECTED_OBSERVATIONS
    assert df["track_id"].nunique() == EXPECTED_TRACKS
    assert not df.duplicated(["frame", "track_id"]).any()
    assert df["frame"].between(1, EXPECTED_FRAMES).all()
    assert df["motion_quality"].isin(ALLOWED_QUALITY).all()

    qualified = df[
        df["motion_quality"] == "LOW_SENSITIVITY"
    ]

    assert qualified["distance_dynamic_m"].notna().all()
    assert qualified["closing_speed_qualified_m_s"].notna().all()
    assert np.isfinite(
        qualified["closing_speed_qualified_m_s"]
    ).all()

    other = df[
        df["motion_quality"] != "LOW_SENSITIVITY"
    ]

    assert other[
        "closing_speed_qualified_m_s"
    ].isna().all()

    assert qualified["window_low_sensitivity"].eq(True).all()

    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened(), "Cannot open output video"

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    assert frames == EXPECTED_FRAMES
    assert (width, height) == (1920, 1080)
    assert abs(fps - 29.97) < 0.1

    # Check that representative frames can be decoded.
    for frame_number in [1, 150, 650, 1023, 1728]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = cap.read()

        assert ok and frame is not None, (
            f"Cannot decode frame {frame_number}"
        )

    cap.release()

    print("\n--- PHASE 5 VALIDATION ---")
    print("All automated checks passed.")
    print(f"Video frames: {frames}")
    print(f"Video resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Observations: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")

    print("\nMotion quality:")
    print(df["motion_quality"].value_counts().to_string())


if __name__ == "__main__":
    main()
