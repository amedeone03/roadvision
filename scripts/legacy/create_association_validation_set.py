
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# -----------------------------------------------
# 1. Configuration
# -----------------------------------------------

VIDEO_PATH = Path("data/videos/test_drive.mp4")

LATERAL_CSV = Path(
    "runs/association_debug/lateral_results.csv"
)

TRACKING_CSV = Path("runs/integrated_restored.csv")

OUTPUT_DIR = Path("runs/association_validation")
IMAGE_DIR = OUTPUT_DIR / "images"
LABELS_CSV = OUTPUT_DIR / "manual_labels.csv"

SEED = 42

# Keep these tracks out of the independent sample.
EXCLUDED_TRACKS = {2, 8, 125}

# Six observations from each category.
SAMPLE_GROUPS = {
    ("car", "IN_LANE"): 6,
    ("car", "OUT_LANE"): 6,
    ("truck", "IN_LANE"): 6,
    ("truck", "OUT_LANE"): 6,
}

rng = np.random.default_rng(SEED)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------
# 2. Load observations
# -----------------------------------------------

lateral = pd.read_csv(LATERAL_CSV)
tracking = pd.read_csv(TRACKING_CSV)

required_lateral = {
    "frame", "track_id", "class_name",
    "RAW_status", "LATERAL_5_status",
    "lane_left_x", "lane_right_x",
    "contact_x", "contact_y",
}

required_tracking = {
    "frame", "track_id", "x1", "y1", "x2", "y2",
}

if required_lateral - set(lateral.columns):
    raise ValueError("Missing lateral analysis columns")

if required_tracking - set(tracking.columns):
    raise ValueError("Missing tracking columns")

# Join bounding boxes to lateral measurements.
df = lateral.merge(
    tracking[
        ["frame", "track_id", "x1", "y1", "x2", "y2"]
    ],
    on=["frame", "track_id"],
    how="inner",
    validate="one_to_one",
)

if len(df) != len(lateral):
    raise RuntimeError("Some tracked observations were lost")

# Exclude previously investigated tracks.
df = df[
    ~df["track_id"].isin(EXCLUDED_TRACKS)
].copy()

# Restrict to definite baseline classifications.
df = df[
    df["RAW_status"].isin(["IN_LANE", "OUT_LANE"])
].copy()


# -----------------------------------------------
# 3. Stratified sampling
# -----------------------------------------------

selected_groups = []

for (class_name, status), target_count in SAMPLE_GROUPS.items():

    group = df[
        (df["class_name"] == class_name)
        & (df["RAW_status"] == status)
    ].copy()

    # Randomize reproducibly.
    if not group.empty:
        group = group.iloc[
            rng.permutation(len(group))
        ]

    chosen = []
    used_frames = set()
    used_tracks = set()

    # First pass: favor different track IDs.
    for row in group.itertuples(index=False):

        if len(chosen) >= target_count:
            break

        if row.frame in used_frames:
            continue

        if row.track_id in used_tracks:
            continue

        chosen.append(row._asdict())
        used_frames.add(row.frame)
        used_tracks.add(row.track_id)

    # Second pass: allow repeated tracks.
    for row in group.itertuples(index=False):

        if len(chosen) >= target_count:
            break

        if row.frame in used_frames:
            continue

        chosen.append(row._asdict())
        used_frames.add(row.frame)

    selected_groups.extend(chosen)

selected = pd.DataFrame(selected_groups)

if selected.empty:
    raise RuntimeError("No validation observations selected")

selected = selected.sort_values(
    ["frame", "track_id"]
).reset_index(drop=True)

print("\n--- ASSOCIATION VALIDATION DATASET ---")
print(f"Selected observations: {len(selected)}")
print(f"Unique tracks: {selected['track_id'].nunique()}")
print(f"Unique frames: {selected['frame'].nunique()}")

print("\nSample distribution:")
print(
    selected.groupby(
        ["class_name", "RAW_status"]
    ).size()
)

changed = (
    selected["RAW_status"]
    != selected["LATERAL_5_status"]
).sum()

print(f"\nBaseline vs 5% disagreements: {changed}")


# -----------------------------------------------
# 4. Extract annotated images
# -----------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video")

review_rows = []

try:

    for index, obj in selected.iterrows():

        frame_number = int(obj["frame"])
        track_id = int(obj["track_id"])

        # CSV frames are 1-based.
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1,
        )

        success, frame = cap.read()

        if not success:
            print(f"Skipping frame {frame_number}")
            continue

        x1 = int(round(obj["x1"]))
        y1 = int(round(obj["y1"]))
        x2 = int(round(obj["x2"]))
        y2 = int(round(obj["y2"]))

        contact_x = int(round(obj["contact_x"]))
        contact_y = int(round(obj["contact_y"]))

        # Highlight the vehicle being reviewed.
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3,
        )

        cv2.circle(
            frame,
            (contact_x, contact_y),
            7,
            (0, 255, 255),
            -1,
        )

        # Show estimated lane boundaries at the
        # contact point's image height.
        if (
            np.isfinite(obj["lane_left_x"])
            and np.isfinite(obj["lane_right_x"])
        ):

            left_x = int(round(obj["lane_left_x"]))
            right_x = int(round(obj["lane_right_x"]))

            cv2.circle(
                frame,
                (left_x, contact_y),
                7,
                (0, 255, 0),
                -1,
            )

            cv2.circle(
                frame,
                (right_x, contact_y),
                7,
                (255, 0, 0),
                -1,
            )

            cv2.line(
                frame,
                (left_x, contact_y),
                (right_x, contact_y),
                (255, 255, 255),
                1,
            )

        # Display identifying information only.
        # Hide predicted lane status during review.
        cv2.putText(
            frame,
            f"Frame {frame_number} | Track {track_id}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            f"Vehicle: {obj['class_name']}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        filename = (
            f"sample_{index + 1:02d}_"
            f"frame_{frame_number}_track_{track_id}.jpg"
        )

        image_path = IMAGE_DIR / filename

        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(
                f"Could not save {image_path}"
            )

        review_rows.append({
            "sample_id": index + 1,
            "image": str(image_path),
            "frame": frame_number,
            "track_id": track_id,
            "class_name": obj["class_name"],
            "manual_label": "",
            "notes": "",
            "RAW_status": obj["RAW_status"],
            "LATERAL_5_status": obj["LATERAL_5_status"],
        })

finally:
    cap.release()


# -----------------------------------------------
# 5. Save review sheet
# -----------------------------------------------

review_df = pd.DataFrame(review_rows)

review_df.to_csv(
    LABELS_CSV,
    index=False,
)

print("\n--- OUTPUT ---")
print(f"Images saved to: {IMAGE_DIR}")
print(f"Review sheet: {LABELS_CSV}")

print("\nManual labels:")
print("IN_LANE  - vehicle occupies the ego lane")
print("OUT_LANE - vehicle occupies another lane")
print("UNSURE   - position cannot be judged reliably")

print("\nValidation dataset created!")
