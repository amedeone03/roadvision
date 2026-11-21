
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

DATASET = Path("data/evaluation/roadvision_v1")

MANIFEST = DATASET / "frame_manifest.csv"
ANNOTATIONS = DATASET / "annotations.csv"
REVIEW_STATUS = DATASET / "review_status.csv"

PREDICTIONS = Path(
    "runs/phase6/final/phase6_results.csv"
)

OUTPUT = Path("runs/phase7/lane_association")

IOU_THRESHOLD = 0.50

LANE_LABELS = [
    "IN_LANE",
    "OUT_LANE",
    "UNKNOWN",
]

KNOWN_LANES = {"IN_LANE", "OUT_LANE"}
BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


# ==================================================
# BOUNDING-BOX MATCHING
# ==================================================

def calculate_iou(a, b):
    """
    Calculate intersection over union between
    two bounding boxes in xyxy format.
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, a[2] - a[0])
        * max(0.0, a[3] - a[1])
    )
    area_b = (
        max(0.0, b[2] - b[0])
        * max(0.0, b[3] - b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_objects(gt_objects, predictions):
    """
    Greedy one-to-one matching by descending IoU.

    Match only objects with the same class and
    sufficient overlap.

    Lane labels are deliberately NOT used for
    matching, since they are what we evaluate.
    """
    candidates = []

    for gi, gt in enumerate(gt_objects):
        for pi, pred in enumerate(predictions):

            if gt["class_name"] != pred["class_name"]:
                continue

            iou = calculate_iou(
                gt["box"],
                pred["box"],
            )

            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in candidates:

        if gi in matched_gt or pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)

        matches.append((gi, pi, iou))

    return matches, matched_gt, matched_pred


# ==================================================
# LOAD DATA
# ==================================================

def load_data():
    manifest = pd.read_csv(MANIFEST)
    annotations = pd.read_csv(ANNOTATIONS)
    status = pd.read_csv(REVIEW_STATUS)
    predictions = pd.read_csv(PREDICTIONS)

    if not manifest["frame"].is_unique:
        raise ValueError("Duplicate frames in manifest")

    reviewed_frames = set(
        status.loc[
            status["reviewed"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"]),
            "frame",
        ].astype(int)
    )

    manifest = manifest[
        manifest["frame"].isin(reviewed_frames)
    ].copy()

    if manifest.empty:
        raise ValueError("No reviewed frames found")

    annotations = annotations[
        annotations["frame"].isin(manifest["frame"])
    ].copy()

    predictions = predictions[
        predictions["frame"].isin(manifest["frame"])
    ].copy()

    for name, df in [
        ("ground truth", annotations),
        ("predictions", predictions),
    ]:
        if df[BOX_COLUMNS].isna().any().any():
            raise ValueError(
                f"Missing box coordinates in {name}"
            )

        if (
            (df["x2"] <= df["x1"]).any()
            or (df["y2"] <= df["y1"]).any()
        ):
            raise ValueError(
                f"Invalid boxes in {name}"
            )

    if annotations.duplicated(
        ["frame", "object_id"]
    ).any():
        raise ValueError(
            "Duplicate ground-truth object IDs"
        )

    if predictions.duplicated(
        ["frame", "track_id"]
    ).any():
        raise ValueError(
            "Duplicate track IDs within frames"
        )

    for column, df in [
        ("lane_gt", annotations),
        ("lane_status", predictions),
    ]:
        invalid = set(
            df[column].dropna()
        ) - set(LANE_LABELS)

        if invalid or df[column].isna().any():
            raise ValueError(
                f"Invalid values in {column}: {invalid}"
            )

    return manifest, annotations, predictions


# ==================================================
# EVALUATION METRICS
# ==================================================

def calculate_lane_metrics(matches):
    """
    Evaluate only matched objects with known
    ground-truth lane membership.

    UNKNOWN predictions count against strict
    accuracy, but are excluded from selective
    accuracy.
    """
    known = matches[
        matches["lane_gt"].isin(KNOWN_LANES)
    ]

    total = len(known)

    definite = known[
        known["lane_pred"].isin(KNOWN_LANES)
    ]

    correct_definite = int(
        (
            definite["lane_gt"]
            == definite["lane_pred"]
        ).sum()
    )

    strict_accuracy = (
        correct_definite / total
        if total else np.nan
    )

    coverage = (
        len(definite) / total
        if total else np.nan
    )

    selective_accuracy = (
        correct_definite / len(definite)
        if len(definite) else np.nan
    )

    return {
        "matched_objects": len(matches),
        "known_gt_objects": total,
        "gt_unknown_objects": len(matches) - total,
        "definite_predictions": len(definite),
        "unknown_predictions_on_known_gt": (
            total - len(definite)
        ),
        "correct_definite": correct_definite,
        "strict_accuracy": strict_accuracy,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "mean_matching_iou": (
            matches["matching_iou"].mean()
            if len(matches) else np.nan
        ),
    }


# ==================================================
# MAIN
# ==================================================

def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest, annotations, predictions = load_data()

    matched_rows = []
    unmatched_rows = []
    frame_rows = []

    for record in manifest.itertuples(index=False):
        frame = int(record.frame)
        subset = record.subset

        gt_df = annotations[
            annotations["frame"] == frame
        ]

        pred_df = predictions[
            predictions["frame"] == frame
        ]

        gt_objects = []

        for gt in gt_df.itertuples(index=False):
            direction = getattr(
                gt, "travel_direction", ""
            )

            gt_objects.append({
                "object_id": int(gt.object_id),
                "class_name": gt.class_name,
                "lane_gt": gt.lane_gt,
                "travel_direction": direction,
                "box": [
                    float(gt.x1),
                    float(gt.y1),
                    float(gt.x2),
                    float(gt.y2),
                ],
            })

        pred_objects = []

        for pred in pred_df.itertuples(index=False):
            pred_objects.append({
                "track_id": int(pred.track_id),
                "class_name": pred.class_name,
                "lane_pred": pred.lane_status,
                "box": [
                    float(pred.x1),
                    float(pred.y1),
                    float(pred.x2),
                    float(pred.y2),
                ],
            })

        matches, used_gt, used_pred = match_objects(
            gt_objects,
            pred_objects,
        )

        for gi, pi, iou in matches:
            gt = gt_objects[gi]
            pred = pred_objects[pi]

            matched_rows.append({
                "frame": frame,
                "subset": subset,
                "object_id": gt["object_id"],
                "track_id": pred["track_id"],
                "class_name": gt["class_name"],
                "lane_gt": gt["lane_gt"],
                "lane_pred": pred["lane_pred"],
                "travel_direction": gt[
                    "travel_direction"
                ],
                "matching_iou": iou,
                "correct": (
                    gt["lane_gt"] == pred["lane_pred"]
                ),
            })

        for gi, gt in enumerate(gt_objects):
            if gi not in used_gt:
                unmatched_rows.append({
                    "frame": frame,
                    "subset": subset,
                    "type": "unmatched_ground_truth",
                    "object_id": gt["object_id"],
                    "track_id": "",
                    "class_name": gt["class_name"],
                })

        for pi, pred in enumerate(pred_objects):
            if pi not in used_pred:
                unmatched_rows.append({
                    "frame": frame,
                    "subset": subset,
                    "type": "unmatched_prediction",
                    "object_id": "",
                    "track_id": pred["track_id"],
                    "class_name": pred["class_name"],
                })

        frame_rows.append({
            "frame": frame,
            "subset": subset,
            "ground_truth_objects": len(gt_objects),
            "tracked_predictions": len(pred_objects),
            "matched_objects": len(matches),
            "unmatched_ground_truth": (
                len(gt_objects) - len(used_gt)
            ),
            "unmatched_predictions": (
                len(pred_objects) - len(used_pred)
            ),
        })

        print(
            f"Frame {frame:04d} | {subset:9s} | "
            f"GT={len(gt_objects)} "
            f"Tracked={len(pred_objects)} "
            f"Matched={len(matches)}"
        )

    # ------------------------------------------
    # Export object-level results
    # ------------------------------------------

    matches_df = pd.DataFrame(
        matched_rows,
        columns=[
            "frame", "subset", "object_id",
            "track_id", "class_name", "lane_gt",
            "lane_pred", "travel_direction",
            "matching_iou", "correct",
        ],
    )

    unmatched_df = pd.DataFrame(
        unmatched_rows,
        columns=[
            "frame", "subset", "type",
            "object_id", "track_id",
            "class_name",
        ],
    )

    frames_df = pd.DataFrame(frame_rows)

    matches_df.to_csv(
        OUTPUT / "matched_lane_objects.csv",
        index=False,
    )

    unmatched_df.to_csv(
        OUTPUT / "unmatched_objects.csv",
        index=False,
    )

    frames_df.to_csv(
        OUTPUT / "frame_coverage.csv",
        index=False,
    )

    # ------------------------------------------
    # Per-subset metrics and confusion matrices
    # ------------------------------------------

    summary_rows = []

    for subset in ["random", "challenge"]:
        subset_matches = matches_df[
            matches_df["subset"] == subset
        ]

        subset_frames = frames_df[
            frames_df["subset"] == subset
        ]

        if subset_frames.empty:
            continue

        metrics = calculate_lane_metrics(
            subset_matches
        )

        metrics.update({
            "subset": subset,
            "frames": len(subset_frames),
            "total_ground_truth": int(
                subset_frames[
                    "ground_truth_objects"
                ].sum()
            ),
            "unmatched_ground_truth": int(
                subset_frames[
                    "unmatched_ground_truth"
                ].sum()
            ),
            "unmatched_predictions": int(
                subset_frames[
                    "unmatched_predictions"
                ].sum()
            ),
        })

        summary_rows.append(metrics)

        # Only GT labels with known lane membership.
        known = subset_matches[
            subset_matches["lane_gt"].isin(
                KNOWN_LANES
            )
        ]

        confusion = pd.crosstab(
            known["lane_gt"],
            known["lane_pred"],
        )

        confusion = confusion.reindex(
            index=["IN_LANE", "OUT_LANE"],
            columns=LANE_LABELS,
            fill_value=0,
        )

        confusion.to_csv(
            OUTPUT / f"confusion_{subset}.csv"
        )

        print(f"\n{subset.upper()} CONFUSION MATRIX")
        print(confusion.to_string())

        # Optional direction-specific subset.
        if "travel_direction" in known.columns:
            direction = (
                known["travel_direction"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.strip()
            )

            same_direction = known[
                direction.isin([
                    "same",
                    "same_direction",
                ])
            ]

            if not same_direction.empty:
                direction_metrics = (
                    calculate_lane_metrics(
                        same_direction
                    )
                )

                print(
                    "\nSame-direction matched-object "
                    f"metrics ({subset}):"
                )
                print(direction_metrics)

    # ------------------------------------------
    # Final summary
    # ------------------------------------------

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT / "summary.csv",
        index=False,
    )

    print("\n--- PHASE 7.3 LANE EVALUATION ---")
    print("Matching IoU threshold:", IOU_THRESHOLD)

    columns = [
        "subset",
        "frames",
        "total_ground_truth",
        "matched_objects",
        "unmatched_ground_truth",
        "strict_accuracy",
        "coverage",
        "selective_accuracy",
    ]

    print(
        summary[columns]
        .round(4)
        .to_string(index=False)
    )

    if not (
        annotations.get(
            "travel_direction",
            pd.Series(dtype=str)
        )
        .astype(str)
        .str.lower()
        .isin(["same", "same_direction"])
        .any()
    ):
        print(
            "\nWARNING: No same-direction labels found. "
            "Lane metrics include all annotated "
            "travel directions and are exploratory."
        )

    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    main()
