
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO = Path("data/videos/test_drive.mp4")

PREDICTIONS = Path(
    "runs/phase6/final/phase6_results.csv"
)

EVENTS = Path(
    "runs/phase6/stabilized_approach_events.csv"
)

OUTPUT = Path("data/evaluation/roadvision_v1")
IMAGE_DIR = OUTPUT / "images"
MANIFEST = OUTPUT / "frame_manifest.csv"

SEED = 42
RANDOM_FRAMES = 60
CHALLENGE_FRAMES = 20
NUM_SEGMENTS = 6

# Used previously for VP development.
PREVIOUS_VP_FRAMES = [
    150, 250, 450, 650,
    850, 1023, 1200, 1450,
]

EXCLUSION_RADIUS = 5


# ==========================================
# SAMPLE FRAMES
# ==========================================

def select_frames(total_frames, predictions, events):
    rng = np.random.default_rng(SEED)

    excluded = {
        frame
        for center in PREVIOUS_VP_FRAMES
        for frame in range(
            max(1, center - EXCLUSION_RADIUS),
            min(total_frames, center + EXCLUSION_RADIUS) + 1,
        )
    }

    eligible = set(range(1, total_frames + 1)) - excluded

    # Random sample spread over the full video.
    segments = np.array_split(
        np.arange(1, total_frames + 1),
        NUM_SEGMENTS,
    )

    if RANDOM_FRAMES % NUM_SEGMENTS != 0:
        raise ValueError(
            "Random frame count must divide evenly "
            "across timeline segments"
        )

    per_segment = RANDOM_FRAMES // NUM_SEGMENTS

    random_sample = []

    for segment in segments:
        candidates = [
            int(f) for f in segment if int(f) in eligible
        ]

        if len(candidates) < per_segment:
            raise ValueError(
                "Not enough eligible frames in a segment"
            )

        selected = rng.choice(
            candidates,
            size=per_segment,
            replace=False,
        )

        random_sample.extend(
            int(f) for f in selected
        )

    selected_set = set(random_sample)

    # --------------------------------------
    # Challenge candidates
    # --------------------------------------

    challenge = []
    reasons = {}

    def add_candidate(frame, reason):
        frame = int(frame)

        if (
            frame in eligible
            and frame not in selected_set
            and frame not in challenge
            and len(challenge) < CHALLENGE_FRAMES
        ):
            challenge.append(frame)
            reasons[frame] = reason

    # 1. Representative frames from events.
    for row in events.itertuples(index=False):
        add_candidate(
            row.confirmation_frame,
            "confirmed_approach_event",
        )

    # 2. Frames with many tracked objects.
    density = (
        predictions.groupby("frame")
        .size()
        .sort_values(ascending=False)
    )

    for frame in density.index:
        if len(challenge) >= 14:
            break
        add_candidate(frame, "high_tracking_density")

    # 3. Frames with uncertain lane association.
    uncertain = (
        predictions[
            predictions["lane_status"] == "UNKNOWN"
        ]
        .groupby("frame")
        .size()
        .sort_values(ascending=False)
    )

    for frame in uncertain.index:
        add_candidate(frame, "uncertain_lane_association")

    # Fallback, if needed.
    remaining = sorted(
        eligible - selected_set - set(challenge)
    )
    rng.shuffle(remaining)

    for frame in remaining:
        add_candidate(frame, "additional_challenge_candidate")

    if len(challenge) != CHALLENGE_FRAMES:
        raise RuntimeError(
            "Could not select the required challenge frames"
        )

    records = [
        {
            "frame": frame,
            "subset": "random",
            "selection_reason": "timeline_stratified_random",
        }
        for frame in random_sample
    ]

    records += [
        {
            "frame": frame,
            "subset": "challenge",
            "selection_reason": reasons[frame],
        }
        for frame in challenge
    ]

    manifest = pd.DataFrame(records)

    assert len(manifest) == (
        RANDOM_FRAMES + CHALLENGE_FRAMES
    )
    assert manifest["frame"].is_unique

    return manifest.sort_values(
        "frame"
    ).reset_index(drop=True)


# ==========================================
# EXTRACT ORIGINAL FRAMES
# ==========================================

def extract_frames(manifest):
    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    try:
        for frame_number in manifest["frame"]:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_number) - 1,
            )

            success, image = cap.read()

            if not success:
                raise RuntimeError(
                    f"Cannot read frame {frame_number}"
                )

            path = (
                IMAGE_DIR
                / f"frame_{int(frame_number):04d}.jpg"
            )

            if not cv2.imwrite(str(path), image):
                raise RuntimeError(
                    f"Could not save {path}"
                )

    finally:
        cap.release()


# ==========================================
# MAIN
# ==========================================

def main():
    if MANIFEST.exists():
        raise FileExistsError(
            f"Dataset already exists: {MANIFEST}. "
            "Preserve the existing selection rather "
            "than silently resampling it."
        )

    predictions = pd.read_csv(PREDICTIONS)
    events = pd.read_csv(EVENTS)

    required = {"frame", "lane_status"}
    if not required.issubset(predictions.columns):
        raise ValueError(
            "Prediction CSV is missing required columns"
        )

    if "confirmation_frame" not in events.columns:
        raise ValueError(
            "Event CSV is missing confirmation_frame"
        )

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    cap.release()

    manifest = select_frames(
        total_frames,
        predictions,
        events,
    )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    extract_frames(manifest)

    manifest["image_file"] = manifest[
        "frame"
    ].apply(
        lambda f: f"images/frame_{int(f):04d}.jpg"
    )

    manifest.to_csv(MANIFEST, index=False)

    print("\n--- ROADVISION EVALUATION DATASET ---")

    print("Video frames:", total_frames)
    print("Selected frames:", len(manifest))

    print("\nSubsets:")
    print(
        manifest["subset"]
        .value_counts()
        .to_string()
    )

    print("\nSelection reasons:")
    print(
        manifest["selection_reason"]
        .value_counts()
        .to_string()
    )

    print("\nOutput:")
    print(OUTPUT)

    print("\nDataset extraction complete!")


if __name__ == "__main__":
    main()
