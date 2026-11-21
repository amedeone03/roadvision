
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from scripts.evaluate_detection import (
    calculate_iou,
    match_objects,
    MODEL_PATH,
    CONF_THRESHOLD,
    IMAGE_SIZE,
    TARGET_CLASSES,
)


DATASET = Path("data/evaluation/roadvision_v1")
OUTPUT = Path("runs/phase7/detection/error_review")

# Prioritized frames from our error audit.
FRAMES = [
    23, 26, 70, 114, 115,
    399, 552, 236, 285,
]


def draw_box(image, box, color, label):
    x1, y1, x2, y2 = map(
        lambda value: int(round(value)),
        box,
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    cv2.putText(
        image,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(
        DATASET / "annotations.csv"
    )

    manifest = pd.read_csv(
        DATASET / "frame_manifest.csv"
    )

    reviewed = pd.read_csv(
        DATASET / "review_status.csv"
    )

    reviewed_frames = set(
        reviewed.loc[
            reviewed["reviewed"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"]),
            "frame",
        ].astype(int)
    )

    model = YOLO(MODEL_PATH)

    for frame in FRAMES:
        if frame not in reviewed_frames:
            raise ValueError(
                f"Frame {frame} has not been reviewed"
            )

        record = manifest[
            manifest["frame"] == frame
        ]

        if len(record) != 1:
            raise ValueError(
                f"Missing or duplicated manifest frame {frame}"
            )

        image_path = (
            DATASET / record.iloc[0]["image_file"]
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Cannot read {image_path}"
            )

        gt_rows = annotations[
            annotations["frame"] == frame
        ]

        ground_truth = [
            {
                "class_name": row.class_name,
                "object_id": int(row.object_id),
                "box": [
                    float(row.x1),
                    float(row.y1),
                    float(row.x2),
                    float(row.y2),
                ],
            }
            for row in gt_rows.itertuples(index=False)
        ]

        result = model.predict(
            image,
            conf=CONF_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]

        predictions = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]

            if class_name not in TARGET_CLASSES:
                continue

            predictions.append({
                "class_name": class_name,
                "confidence": float(box.conf.item()),
                "box": (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
            })

        matches, matched_pred, matched_gt = (
            match_objects(predictions, ground_truth)
        )

        # Blue = manually annotated object.
        for gt in ground_truth:
            draw_box(
                image,
                gt["box"],
                (255, 100, 0),
                (
                    f"GT #{gt['object_id']} "
                    f"{gt['class_name']}"
                ),
            )

        # Green = matched prediction.
        # Red = unmatched prediction.
        for index, pred in enumerate(predictions):
            is_matched = index in matched_pred

            color = (
                (0, 220, 0)
                if is_matched
                else (0, 0, 255)
            )

            label = (
                "TP" if is_matched else "FP"
            )

            draw_box(
                image,
                pred["box"],
                color,
                (
                    f"{label} {pred['class_name']} "
                    f"{pred['confidence']:.2f}"
                ),
            )

        # Highlight missed ground truth.
        for index, gt in enumerate(ground_truth):
            if index not in matched_gt:
                draw_box(
                    image,
                    gt["box"],
                    (255, 0, 255),
                    f"FN #{gt['object_id']}",
                )

        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp

        text = (
            f"Frame {frame} | "
            f"TP={tp} FP={fp} FN={fn}"
        )

        cv2.putText(
            image,
            text,
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        output_path = (
            OUTPUT / f"frame_{frame:04d}.jpg"
        )

        if not cv2.imwrite(
            str(output_path),
            image,
        ):
            raise RuntimeError(
                f"Could not save {output_path}"
            )

        print(
            f"Frame {frame:04d}: "
            f"TP={tp} FP={fp} FN={fn}"
        )

    print("\nReview images saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()