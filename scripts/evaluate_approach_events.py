
from pathlib import Path

import cv2
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

VIDEO = Path("data/videos/test_drive.mp4")

EVENTS = Path(
    "runs/phase6/stabilized_approach_events.csv"
)

OBSERVATIONS = Path(
    "runs/phase6/final/phase6_results.csv"
)

OUTPUT = Path("runs/phase7/approach_events")
CLIPS = OUTPUT / "clips"

BEFORE_FRAMES = 20
AFTER_FRAMES = 20

OUTPUT_WIDTH = 960

REVIEW_COLUMNS = [
    "event_id",
    "track_id",
    "start_frame",
    "end_frame",
    "apparent_approach",
    "ego_lane_correct",
    "tracking_stable",
    "review_notes",
]

# Manual review options:
# apparent_approach: YES / NO / UNCERTAIN
# ego_lane_correct: YES / NO / UNCERTAIN
# tracking_stable: YES / NO / UNCERTAIN


# ==================================================
# COLUMN HELPERS
# ==================================================

def find_column(df, candidates, required=True):
    """
    Accept common alternatives for column names.
    """
    for name in candidates:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            f"Expected one of {candidates}; "
            f"available columns: {list(df.columns)}"
        )

    return None


def get_optional(row, column, default="N/A"):
    if column not in row.index:
        return default

    value = row[column]

    return default if pd.isna(value) else str(value)


# ==================================================
# VIDEO OVERLAY
# ==================================================

def draw_label(image, text, x, y, color):
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame,
    frame_number,
    track_id,
    observations,
    event_start,
    confirmed_frame,
    event_end,
):
    image = frame.copy()

    if frame_number < event_start:
        phase = "BEFORE EVENT"
    elif frame_number < confirmed_frame:
        phase = "PENDING / PRE-CONFIRMATION"
    elif frame_number <= event_end:
        phase = "CONFIRMED EPISODE"
    else:
        phase = "AFTER EVENT"

    # These labels describe the episode timeline,
    # not necessarily the online state at this frame.
    draw_label(
        image,
        f"Frame {frame_number} | Track {track_id}",
        25,
        42,
        (255, 255, 255),
    )

    draw_label(
        image,
        phase,
        25,
        76,
        (255, 255, 255),
    )

    selected = observations[
        (observations["frame"] == frame_number)
        & (observations["track_id"] == track_id)
    ]

    if selected.empty:
        draw_label(
            image,
            "Target track: NO OBSERVATION",
            25,
            112,
            (0, 170, 255),
        )
        return image

    row = selected.iloc[0]

    coords = [
        int(round(float(row[c])))
        for c in ["x1", "y1", "x2", "y2"]
    ]

    x1, y1, x2, y2 = coords

    state = get_optional(
        row, "event_state"
    )

    lane = get_optional(
        row, "lane_status"
    )

    motion = get_optional(
        row, "approach_state"
    )

    # Green for online-confirmed state;
    # yellow for other available observations.
    color = (
        (0, 220, 0)
        if state == "CONFIRMED"
        else (0, 230, 255)
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    draw_label(
        image,
        f"ID {track_id} | Lane: {lane}",
        x1,
        max(140, y1 - 12),
        color,
    )

    draw_label(
        image,
        f"Event: {state} | Motion: {motion}",
        25,
        112,
        (255, 255, 255),
    )

    return image


# ==================================================
# EVENT DATA
# ==================================================

def load_events_and_observations():
    if not EVENTS.exists():
        raise FileNotFoundError(EVENTS)

    if not OBSERVATIONS.exists():
        raise FileNotFoundError(OBSERVATIONS)

    events = pd.read_csv(EVENTS)
    observations = pd.read_csv(OBSERVATIONS)

    track_col = find_column(
        events, ["track_id"]
    )

    start_col = find_column(
        events, ["start_frame"]
    )

    end_col = find_column(
        events, ["end_frame", "last_positive_frame"]
    )

    confirmation_col = find_column(
        events,
        [
            "confirmed_frame",
            "confirmation_frame",
            "first_confirmed_frame",
        ],
        required=False,
    )

    necessary = {
        "frame", "track_id", "x1", "y1",
        "x2", "y2",
    }

    missing = necessary - set(observations.columns)

    if missing:
        raise ValueError(
            f"Missing observation columns: {missing}"
        )

    observations["frame"] = (
        observations["frame"].astype(int)
    )

    observations["track_id"] = (
        observations["track_id"].astype(int)
    )

    if observations.duplicated(
        ["frame", "track_id"]
    ).any():
        raise ValueError(
            "Duplicate track observations per frame"
        )

    records = []

    events = events.sort_values(
        [start_col, track_col]
    ).reset_index(drop=True)

    for index, row in events.iterrows():
        track_id = int(row[track_col])
        start = int(row[start_col])
        end = int(row[end_col])

        if start < 1 or end < start:
            raise ValueError(
                f"Invalid episode frames for track {track_id}"
            )

        # Prefer confirmation explicitly saved by Phase 6.
        if (
            confirmation_col is not None
            and pd.notna(row[confirmation_col])
        ):
            confirmed = int(row[confirmation_col])
        else:
            # Fallback: derive from online state.
            if "event_state" not in observations.columns:
                raise ValueError(
                    "Cannot determine confirmation frame"
                )

            candidates = observations[
                (observations["track_id"] == track_id)
                & (observations["frame"] >= start)
                & (observations["frame"] <= end)
                & (
                    observations["event_state"]
                    == "CONFIRMED"
                )
            ]

            if candidates.empty:
                raise ValueError(
                    f"No confirmation found for track "
                    f"{track_id}, frames {start}-{end}"
                )

            confirmed = int(
                candidates["frame"].min()
            )

        if not start <= confirmed <= end:
            raise ValueError(
                f"Confirmation outside episode "
                f"for track {track_id}"
            )

        records.append({
            "event_id": index + 1,
            "track_id": track_id,
            "start_frame": start,
            "confirmed_frame": confirmed,
            "end_frame": end,
        })

    return pd.DataFrame(records), observations


# ==================================================
# EXPORT CLIPS
# ==================================================

def create_clip(
    cap,
    event,
    observations,
    fps,
    video_frames,
):
    event_id = int(event["event_id"])
    track_id = int(event["track_id"])

    start = int(event["start_frame"])
    confirmed = int(event["confirmed_frame"])
    end = int(event["end_frame"])

    clip_start = max(
        1, start - BEFORE_FRAMES
    )

    clip_end = min(
        video_frames, end + AFTER_FRAMES
    )

    output_path = (
        CLIPS / f"event_{event_id:02d}"
        f"_track_{track_id}.mp4"
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        clip_start - 1,
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_height = int(
        round(height * OUTPUT_WIDTH / width)
    )

    # MP4 writer compatibility.
    output_height -= output_height % 2

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUTPUT_WIDTH, output_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create {output_path}"
        )

    try:
        for frame_number in range(
            clip_start, clip_end + 1
        ):
            success, frame = cap.read()

            if not success:
                raise RuntimeError(
                    f"Cannot read video frame {frame_number}"
                )

            image = annotate_frame(
                frame,
                frame_number,
                track_id,
                observations,
                start,
                confirmed,
                end,
            )

            image = cv2.resize(
                image,
                (OUTPUT_WIDTH, output_height),
            )

            writer.write(image)

    finally:
        writer.release()

    return output_path


# ==================================================
# MAIN
# ==================================================

def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )
    CLIPS.mkdir(
        parents=True,
        exist_ok=True,
    )

    events, observations = (
        load_events_and_observations()
    )

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {VIDEO}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )
    video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0 or video_frames <= 0:
        cap.release()
        raise RuntimeError(
            "Invalid video metadata"
        )

    results = []

    try:
        for _, event in events.iterrows():

            start = int(event["start_frame"])
            confirmed = int(event["confirmed_frame"])
            end = int(event["end_frame"])

            if end > video_frames:
                raise ValueError(
                    "Episode exceeds video length"
                )

            clip_path = create_clip(
                cap,
                event,
                observations,
                fps,
                video_frames,
            )

            delay_frames = confirmed - start

            episode_frames = end - start + 1

            results.append({
                "event_id": int(event["event_id"]),
                "track_id": int(event["track_id"]),
                "start_frame": start,
                "confirmed_frame": confirmed,
                "end_frame": end,
                "confirmation_delay_frames": (
                    delay_frames
                ),
                "confirmation_delay_seconds": (
                    delay_frames / fps
                ),
                "episode_frames": episode_frames,
                "episode_span_seconds": (
                    episode_frames / fps
                ),
                "clip_file": str(clip_path),
            })

            print(
                f"Event {int(event['event_id']):02d} | "
                f"Track {int(event['track_id']):3d} | "
                f"Frames {start}-{end} | "
                f"Confirmed {confirmed} | "
                f"Delay {delay_frames} frames"
            )

    finally:
        cap.release()

    metrics = pd.DataFrame(results)

    metrics.to_csv(
        OUTPUT / "event_metrics.csv",
        index=False,
    )

    # Create review template without overwriting
    # previous human judgments.
    review_file = (
        OUTPUT / "manual_event_review.csv"
    )

    if not review_file.exists():

        review = events[
            [
                "event_id",
                "track_id",
                "start_frame",
                "end_frame",
            ]
        ].copy()

        review["apparent_approach"] = ""
        review["ego_lane_correct"] = ""
        review["tracking_stable"] = ""
        review["review_notes"] = ""

        review[REVIEW_COLUMNS].to_csv(
            review_file,
            index=False,
        )

    print("\n--- PHASE 7.5 EVENT EVALUATION ---")
    print("Completed episodes:", len(metrics))
    print(
        "Unique tracks:",
        metrics["track_id"].nunique(),
    )

    print(
        "Mean confirmation delay (frames):",
        round(
            metrics[
                "confirmation_delay_frames"
            ].mean(),
            2,
        ),
    )

    print(
        "Maximum episode length (frames):",
        int(metrics["episode_frames"].max()),
    )

    print("\nSaved:")
    print(OUTPUT / "event_metrics.csv")
    print(review_file)
    print(CLIPS)


if __name__ == "__main__":
    main()
