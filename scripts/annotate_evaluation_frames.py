
from pathlib import Path
import csv

import cv2
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

DATASET = Path("data/evaluation/roadvision_v1")

MANIFEST = DATASET / "frame_manifest.csv"
ANNOTATIONS = DATASET / "annotations.csv"
REVIEW_STATUS = DATASET / "review_status.csv"

# False = annotate the entire evaluation dataset.
PILOT_ONLY = False
PILOT_FRAMES = 5

# False = automatically skip previously reviewed frames.
# Set True later if you want to revisit all images.
REVIEW_ALL = False
REVIEW_FRAMES = []

MAX_WIDTH = 1280
MAX_HEIGHT = 720

WINDOW = "RoadVision Ground Truth"

CLASSES = {
    ord("1"): "car",
    ord("2"): "truck",
    ord("3"): "bus",
    ord("4"): "motorcycle",
    ord("5"): "bicycle",
    ord("6"): "person",
}

LANES = {
    ord("i"): "IN_LANE",
    ord("o"): "OUT_LANE",
    ord("u"): "UNKNOWN",
}

COLUMNS = [
    "frame",
    "object_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "lane_gt",
    "occluded",
    "truncated",
]


# ==================================================
# 2. CSV HELPERS
# ==================================================

def save_csv(path, rows, columns):
    """
    Save CSV atomically using a temporary file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                column: row.get(column, "")
                for column in columns
            })

    temporary.replace(path)


def load_existing(path):
    """
    Load existing CSV data.

    Older annotations may not have occluded
    or truncated fields. Preserve them as
    blank rather than assuming False.
    """

    if not path.exists():
        return []

    df = pd.read_csv(
        path,
        keep_default_na=False,
    )

    if path == ANNOTATIONS:

        for column in ["occluded", "truncated"]:
            if column not in df.columns:
                df[column] = ""

    return df.to_dict("records")


# ==================================================
# 3. DRAWING HELPERS
# ==================================================

def draw_text(
    image,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.48,
):

    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def format_flag(value):
    """
    Display True, False, or ? for metadata
    that has not yet been assessed.
    """

    if value in (True, "True", "true", "1", 1):
        return "Y"

    if value in (False, "False", "false", "0", 0):
        return "N"

    return "?"


# ==================================================
# 4. MAIN ANNOTATION APPLICATION
# ==================================================

def main():

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    manifest = pd.read_csv(MANIFEST)

    # ------------------------------------------
    # Optional pilot mode
    # ------------------------------------------

    if PILOT_ONLY:
        manifest = (
            manifest[
                manifest["subset"] == "random"
            ]
            .head(PILOT_FRAMES)
            .reset_index(drop=True)
        )

    if manifest.empty:
        raise RuntimeError(
            "No images found in the manifest"
        )

    # ------------------------------------------
    # Load existing work
    # ------------------------------------------

    annotations = load_existing(ANNOTATIONS)
    reviewed_rows = load_existing(REVIEW_STATUS)

    reviewed = {
        int(row["frame"])
        for row in reviewed_rows
        if str(row["reviewed"]).lower()
        in ("true", "1")
    }

    # ------------------------------------------
    # Resume from unreviewed frames
    # ------------------------------------------

    if REVIEW_FRAMES:
        manifest = manifest[
            manifest["frame"].isin(REVIEW_FRAMES)
        ].reset_index(drop=True)

    elif not REVIEW_ALL:
        manifest = manifest[
            ~manifest["frame"].isin(reviewed)
        ].reset_index(drop=True)

    print("\n--- ROADVISION ANNOTATION TOOL ---")
    print("Previously reviewed:", len(reviewed))
    print("Remaining frames:", len(manifest))
    print("Existing annotations:", len(annotations))

    print("\nControls:")
    print("1-6: Select class")
    print("I / O / U: Select lane")
    print("C: Toggle occluded")
    print("T: Toggle truncated")
    print("Mouse drag: Draw bounding box")
    print("Z: Undo last box")
    print("N: Save and next frame")
    print("P: Save and previous frame")
    print("Q: Save current edits and quit")

    index = 0
    stop = False

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_AUTOSIZE,
    )

    # ==================================================
    # FRAME LOOP
    # ==================================================

    try:

        while 0 <= index < len(manifest) and not stop:

            record = manifest.iloc[index]

            frame_number = int(record["frame"])

            image_path = (
                DATASET / record["image_file"]
            )

            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(
                    f"Cannot read image: {image_path}"
                )

            height, width = image.shape[:2]

            scale = min(
                MAX_WIDTH / width,
                MAX_HEIGHT / height,
                1.0,
            )

            display_width = int(
                round(width * scale)
            )

            display_height = int(
                round(height * scale)
            )

            # --------------------------------------
            # Existing boxes for this frame
            # --------------------------------------

            boxes = [
                dict(row)
                for row in annotations
                if int(row["frame"]) == frame_number
            ]

            boxes.sort(
                key=lambda row: int(row["object_id"])
            )

            # --------------------------------------
            # Current annotation selections
            # --------------------------------------

            current_class = "car"
            current_lane = "UNKNOWN"

            current_occluded = False
            current_truncated = False

            mouse = {
                "drawing": False,
                "start": None,
                "end": None,
            }

            # ======================================
            # MOUSE CALLBACK
            # ======================================

            def on_mouse(
                event,
                x,
                y,
                flags,
                userdata,
            ):

                if event == cv2.EVENT_LBUTTONDOWN:

                    # Ignore the control bar.
                    if not (
                        0 <= x < display_width
                        and 0 <= y < display_height
                    ):
                        return

                    mouse["drawing"] = True
                    mouse["start"] = (x, y)
                    mouse["end"] = (x, y)

                elif (
                    event == cv2.EVENT_MOUSEMOVE
                    and mouse["drawing"]
                ):

                    mouse["end"] = (
                        max(
                            0,
                            min(x, display_width - 1),
                        ),
                        max(
                            0,
                            min(y, display_height - 1),
                        ),
                    )

                elif (
                    event == cv2.EVENT_LBUTTONUP
                    and mouse["drawing"]
                ):

                    mouse["drawing"] = False

                    sx, sy = mouse["start"]

                    x = max(
                        0,
                        min(x, display_width - 1),
                    )

                    y = max(
                        0,
                        min(y, display_height - 1),
                    )

                    x1, x2 = sorted([sx, x])
                    y1, y2 = sorted([sy, y])

                    # Avoid accidental tiny boxes.
                    if (
                        x2 - x1 < 4
                        or y2 - y1 < 4
                    ):
                        mouse["start"] = None
                        mouse["end"] = None
                        return

                    next_id = (
                        max(
                            (
                                int(box["object_id"])
                                for box in boxes
                            ),
                            default=0,
                        )
                        + 1
                    )

                    boxes.append({
                        "frame": frame_number,
                        "object_id": next_id,
                        "class_name": current_class,
                        "x1": round(x1 / scale, 2),
                        "y1": round(y1 / scale, 2),
                        "x2": round(x2 / scale, 2),
                        "y2": round(y2 / scale, 2),
                        "lane_gt": current_lane,
                        "occluded": current_occluded,
                        "truncated": current_truncated,
                    })

                    mouse["start"] = None
                    mouse["end"] = None

            cv2.setMouseCallback(
                WINDOW,
                on_mouse,
            )

            # ======================================
            # SAVE CURRENT FRAME
            # ======================================

            def persist(mark_reviewed=True):
                nonlocal annotations

                # Preserve other frames.
                annotations = [
                    row
                    for row in annotations
                    if int(row["frame"]) != frame_number
                ]

                # Replace only current-frame data.
                annotations.extend(boxes)

                annotations.sort(
                    key=lambda row: (
                        int(row["frame"]),
                        int(row["object_id"]),
                    )
                )

                save_csv(
                    ANNOTATIONS,
                    annotations,
                    COLUMNS,
                )

                if mark_reviewed:
                    reviewed.add(frame_number)

                status_rows = [
                    {
                        "frame": frame,
                        "reviewed": True,
                    }
                    for frame in sorted(reviewed)
                ]

                save_csv(
                    REVIEW_STATUS,
                    status_rows,
                    ["frame", "reviewed"],
                )

            # ======================================
            # DISPLAY LOOP
            # ======================================

            while True:

                canvas = cv2.resize(
                    image,
                    (
                        display_width,
                        display_height,
                    ),
                )

                # ----------------------------------
                # Draw annotated boxes
                # ----------------------------------

                for box in boxes:

                    p1 = (
                        int(round(
                            float(box["x1"]) * scale
                        )),
                        int(round(
                            float(box["y1"]) * scale
                        )),
                    )

                    p2 = (
                        int(round(
                            float(box["x2"]) * scale
                        )),
                        int(round(
                            float(box["y2"]) * scale
                        )),
                    )

                    cv2.rectangle(
                        canvas,
                        p1,
                        p2,
                        (0, 255, 255),
                        2,
                    )

                    occluded = format_flag(
                        box.get("occluded", "")
                    )

                    truncated = format_flag(
                        box.get("truncated", "")
                    )

                    label = (
                        f"#{int(box['object_id'])} "
                        f"{box['class_name']} "
                        f"{box['lane_gt']} "
                        f"Occ:{occluded} "
                        f"Trunc:{truncated}"
                    )

                    draw_text(
                        canvas,
                        label,
                        p1[0],
                        max(18, p1[1] - 6),
                        color=(0, 255, 255),
                        scale=0.43,
                    )

                # ----------------------------------
                # Current mouse rectangle
                # ----------------------------------

                if (
                    mouse["drawing"]
                    and mouse["start"] is not None
                    and mouse["end"] is not None
                ):

                    cv2.rectangle(
                        canvas,
                        mouse["start"],
                        mouse["end"],
                        (255, 255, 0),
                        2,
                    )

                # ----------------------------------
                # Control bar
                # ----------------------------------

                bar = cv2.copyMakeBorder(
                    canvas,
                    0,
                    124,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(22, 22, 22),
                )

                lines = [
                    (
                        f"Frame {frame_number} | "
                        f"{index + 1}/{len(manifest)} | "
                        f"Boxes: {len(boxes)}"
                    ),
                    (
                        f"Selected: {current_class} | "
                        f"{current_lane} | "
                        f"Occluded: {current_occluded} | "
                        f"Truncated: {current_truncated}"
                    ),
                    (
                        "1 car  2 truck  3 bus  "
                        "4 motorcycle  5 bicycle  6 person"
                    ),
                    (
                        "I/O/U lane | C occluded | "
                        "T truncated | Z undo | "
                        "N next | P previous | Q quit"
                    ),
                ]

                for j, line in enumerate(lines):

                    draw_text(
                        bar,
                        line,
                        12,
                        display_height + 24 + 28*j,
                        scale=0.49,
                    )

                cv2.imshow(
                    WINDOW,
                    bar,
                )

                # ----------------------------------
                # Keyboard controls
                # ----------------------------------

                key = cv2.waitKey(30) & 0xFF

                if key in CLASSES:

                    current_class = CLASSES[key]

                elif key in LANES:

                    current_lane = LANES[key]

                elif key == ord("c"):

                    current_occluded = (
                        not current_occluded
                    )

                elif key == ord("t"):

                    current_truncated = (
                        not current_truncated
                    )

                elif key == ord("z"):

                    if boxes:
                        boxes.pop()

                elif key == ord("n"):

                    persist(mark_reviewed=True)
                    index += 1
                    break

                elif key == ord("p"):

                    persist(mark_reviewed=True)

                    index = max(
                        0,
                        index - 1,
                    )

                    break

                elif key == ord("q"):

                    # Save partial work without
                    # marking this frame reviewed.
                    persist(mark_reviewed=False)

                    stop = True
                    break

    finally:

        cv2.destroyAllWindows()

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    all_manifest = pd.read_csv(MANIFEST)

    total = len(all_manifest)
    completed = len(
        set(all_manifest["frame"]) & reviewed
    )

    print("\n--- ANNOTATION PROGRESS ---")
    print(f"Reviewed frames: {completed}/{total}")
    print(f"Remaining frames: {total - completed}")
    print(f"Object annotations: {len(annotations)}")
    print(f"Annotations saved: {ANNOTATIONS}")
    print(f"Review status saved: {REVIEW_STATUS}")

    print("\nAnnotation session finished.")


if __name__ == "__main__":
    main()
