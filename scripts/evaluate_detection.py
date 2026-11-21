
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

DATASET = Path("data/evaluation/roadvision_v1")
MANIFEST = DATASET / "frame_manifest.csv"
ANNOTATIONS = DATASET / "annotations.csv"
REVIEW_STATUS = DATASET / "review_status.csv"

OUTPUT = Path("runs/phase7/detection")
MODEL_PATH = "yolo11n.pt"

CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.50
IMAGE_SIZE = 640

TARGET_CLASSES = {
    "person", "bicycle", "car",
    "motorcycle", "bus", "truck",
}

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


# ==========================================
# BOUNDING-BOX IOU
# ==========================================

def calculate_iou(box_a, box_b):
    """
    Intersection over Union of two xyxy boxes.
    """

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )
    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ==========================================
# ONE-TO-ONE MATCHING
# ==========================================

def match_objects(predictions, ground_truth):
    """
    Greedy matching by descending IoU.

    A match requires:
      - The same object class
      - IoU >= IOU_MATCH_THRESHOLD

    Every prediction and ground-truth object
    can be matched at most once.
    """

    candidates = []

    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):

            if pred["class_name"] != gt["class_name"]:
                continue

            iou = calculate_iou(
                pred["box"],
                gt["box"],
            )

            if iou >= IOU_MATCH_THRESHOLD:
                candidates.append((iou, pi, gi))

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched_pred = set()
    matched_gt = set()
    matches = []

    for iou, pi, gi in candidates:

        if pi in matched_pred or gi in matched_gt:
            continue

        matched_pred.add(pi)
        matched_gt.add(gi)

        matches.append((pi, gi, iou))

    return matches, matched_pred, matched_gt


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(results):
    tp = int(results["tp"].sum())
    fp = int(results["fp"].sum())
    fn = int(results["fn"].sum())

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "frames": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ==========================================
# LOAD EVALUATION DATASET
# ==========================================

def load_dataset():
    manifest = pd.read_csv(MANIFEST)
    annotations = pd.read_csv(ANNOTATIONS)
    status = pd.read_csv(REVIEW_STATUS)

    reviewed_frames = set(
        status.loc[
            status["reviewed"].astype(str).str.lower()
            .isin(["true", "1"]),
            "frame",
        ].astype(int)
    )

    evaluated = manifest[
        manifest["frame"].isin(reviewed_frames)
    ].copy()

    if evaluated.empty:
        raise RuntimeError(
            "No reviewed frames available"
        )

    if not evaluated["frame"].is_unique:
        raise ValueError(
            "Duplicate frames in evaluation manifest"
        )

    # Only evaluate reviewed frames.
    annotations = annotations[
        annotations["frame"].isin(evaluated["frame"])
    ].copy()

    if annotations[BOX_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing bounding-box coordinates"
        )

    if (
        (annotations["x2"] <= annotations["x1"]).any()
        or (annotations["y2"] <= annotations["y1"]).any()
    ):
        raise ValueError(
            "Invalid ground-truth bounding boxes"
        )

    invalid_classes = (
        set(annotations["class_name"])
        - TARGET_CLASSES
    )

    if invalid_classes:
        raise ValueError(
            f"Unsupported classes: {invalid_classes}"
        )

    return evaluated, annotations


# ==========================================
# MAIN EVALUATION
# ==========================================

def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest, annotations = load_dataset()

    model = YOLO(MODEL_PATH)

    frame_results = []
    match_results = []
    error_results = []

    for row in manifest.itertuples(index=False):

        frame = int(row.frame)
        subset = row.subset

        image = cv2.imread(
            str(DATASET / row.image_file)
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read frame {frame}"
            )

        height, width = image.shape[:2]

        # Load manually labeled objects.
        gt_rows = annotations[
            annotations["frame"] == frame
        ]

        ground_truth = []

        for gt in gt_rows.itertuples(index=False):

            box = [
                float(gt.x1),
                float(gt.y1),
                float(gt.x2),
                float(gt.y2),
            ]

            if not (
                0 <= box[0] < box[2] <= width
                and 0 <= box[1] < box[3] <= height
            ):
                raise ValueError(
                    f"Box outside image in frame {frame}"
                )

            ground_truth.append({
                "class_name": gt.class_name,
                "box": box,
                "object_id": int(gt.object_id),
            })

        # Run YOLO on the ORIGINAL image.
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
                "box": box.xyxy[0].cpu().numpy().tolist(),
                "confidence": float(box.conf.item()),
            })

        # Match predictions and ground truth.
        matches, matched_pred, matched_gt = (
            match_objects(predictions, ground_truth)
        )

        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp

        frame_results.append({
            "frame": frame,
            "subset": subset,
            "ground_truth": len(ground_truth),
            "predictions": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        # Save correct matches.
        for pi, gi, iou in matches:

            match_results.append({
                "frame": frame,
                "subset": subset,
                "class_name": predictions[pi]["class_name"],
                "confidence": predictions[pi]["confidence"],
                "iou": iou,
                "object_id": ground_truth[gi]["object_id"],
            })

        # Save unmatched predictions (FP).
        for pi, pred in enumerate(predictions):

            if pi not in matched_pred:

                error_results.append({
                    "frame": frame,
                    "subset": subset,
                    "error_type": "FP",
                    "class_name": pred["class_name"],
                    "confidence": pred["confidence"],
                    "object_id": "",
                    "box": str(pred["box"]),
                })

        # Save missed ground-truth objects (FN).
        for gi, gt in enumerate(ground_truth):

            if gi not in matched_gt:

                error_results.append({
                    "frame": frame,
                    "subset": subset,
                    "error_type": "FN",
                    "class_name": gt["class_name"],
                    "confidence": np.nan,
                    "object_id": gt["object_id"],
                    "box": str(gt["box"]),
                })

        print(
            f"Frame {frame:04d} | {subset:9s} | "
            f"TP={tp} FP={fp} FN={fn}"
        )

    # ======================================
    # EXPORT RESULTS
    # ======================================

    frames_df = pd.DataFrame(frame_results)
    matches_df = pd.DataFrame(
        match_results,
        columns=[
            "frame", "subset", "class_name",
            "confidence", "iou", "object_id",
        ],
    )
    errors_df = pd.DataFrame(
        error_results,
        columns=[
            "frame", "subset", "error_type",
            "class_name", "confidence",
            "object_id", "box",
        ],
    )

    frames_df.to_csv(
        OUTPUT / "frame_metrics.csv",
        index=False,
    )

    matches_df.to_csv(
        OUTPUT / "matched_detections.csv",
        index=False,
    )

    errors_df.to_csv(
        OUTPUT / "detection_errors.csv",
        index=False,
    )

    summary_rows = []

    for subset in ["random", "challenge"]:

        subset_frames = frames_df[
            frames_df["subset"] == subset
        ]

        if subset_frames.empty:
            continue

        metrics = calculate_metrics(subset_frames)

        subset_matches = matches_df[
            matches_df["subset"] == subset
        ]

        metrics["subset"] = subset
        metrics["mean_matched_iou"] = (
            subset_matches["iou"].mean()
            if not subset_matches.empty
            else np.nan
        )

        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT / "summary.csv",
        index=False,
    )

    # ======================================
    # PRINT SUMMARY
    # ======================================

    print("\n--- PHASE 7.2 DETECTION EVALUATION ---")

    print("\nConfiguration:")
    print("Model:", MODEL_PATH)
    print("Confidence threshold:", CONF_THRESHOLD)
    print("Matching IoU threshold:", IOU_MATCH_THRESHOLD)
    print("Image size:", IMAGE_SIZE)

    print("\nResults:")
    print(
        summary[
            [
                "subset",
                "frames",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_matched_iou",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved results to:", OUTPUT)


if __name__ == "__main__":
    main()
