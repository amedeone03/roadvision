
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO = Path("runs/phase6/final/phase6_output.mp4")
CSV = Path("runs/phase6/final/phase6_results.csv")
EVENTS = Path("runs/phase6/stabilized_approach_events.csv")

EXPECTED_FRAMES = 1728
EXPECTED_OBSERVATIONS = 2656
EXPECTED_TRACKS = 68

VALID_STATES = {
    "NONE", "PENDING", "CONFIRMED", "HELD"
}


def main():
    for path in [VIDEO, CSV, EVENTS]:
        assert path.exists(), f"Missing: {path}"

    df = pd.read_csv(CSV)
    events = pd.read_csv(EVENTS)

    # Structural consistency
    assert len(df) == EXPECTED_OBSERVATIONS
    assert df["track_id"].nunique() == EXPECTED_TRACKS
    assert df["frame"].between(1, EXPECTED_FRAMES).all()
    assert not df.duplicated(["frame", "track_id"]).any()

    # State-machine consistency
    assert df["event_state"].isin(VALID_STATES).all()

    confirmed = df[df["event_state"] == "CONFIRMED"]
    held = df[df["event_state"] == "HELD"]

    assert (confirmed["lane_status"] == "IN_LANE").all()
    assert (
        confirmed["approach_t015"] == "APPROACHING"
    ).all()
    assert confirmed["tau_proxy_s"].notna().all()
    assert (
        np.isfinite(confirmed["tau_proxy_s"])
        & (confirmed["tau_proxy_s"] > 0)
    ).all()

    assert (held["lane_status"] != "OUT_LANE").all()
    assert (held["approach_t015"] != "RECEDING").all()

    # Check episode summaries
    assert not events.empty
    assert (events["start_frame"] <= events["confirmation_frame"]).all()
    assert (events["confirmation_frame"] <= events["end_frame"]).all()
    assert (
        events["duration_frames"]
        == events["end_frame"] - events["start_frame"] + 1
    ).all()

    assert (
        events["positive_frames"] + events["interruption_frames"]
        == events["duration_frames"]
    ).all()

    # Video properties and decoding
    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened(), "Cannot open Phase 6 video"

    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        assert frames == EXPECTED_FRAMES
        assert (width, height) == (1920, 1080)
        assert abs(fps - 29.97) < 0.1

        for frame_number in [
            1, 42, 83, 84, 589, 591, 599, 600, 1728
        ]:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number - 1,
            )
            ok, frame = cap.read()

            assert ok and frame is not None, (
                f"Cannot decode frame {frame_number}"
            )
    finally:
        cap.release()

    print("\n--- PHASE 6 VALIDATION ---")
    print("All integration checks passed.")
    print(f"Video: {frames} frames, {width}x{height}, {fps:.2f} FPS")
    print(f"Observations: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Completed episodes: {len(events)}")

    print("\nEvent states:")
    print(df["event_state"].value_counts().to_string())


if __name__ == "__main__":
    main()
