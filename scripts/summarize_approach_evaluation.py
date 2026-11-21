
from pathlib import Path

import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path("runs/phase7/approach_events")

REVIEW_FILE = ROOT / "manual_event_review.csv"
METRICS_FILE = ROOT / "event_metrics.csv"

OUTPUT_CSV = ROOT / "review_summary.csv"
OUTPUT_REPORT = ROOT / "evaluation_report.md"

REVIEW_FIELDS = [
    "apparent_approach",
    "ego_lane_correct",
    "tracking_stable",
]

VALID_LABELS = {"YES", "NO", "UNCERTAIN"}


# ==================================================
# LOAD AND VALIDATE
# ==================================================

def load_data():
    review = pd.read_csv(
        REVIEW_FILE,
        keep_default_na=False,
    )

    metrics = pd.read_csv(METRICS_FILE)

    required = {"event_id", *REVIEW_FIELDS}

    missing = required - set(review.columns)

    if missing:
        raise ValueError(
            f"Missing review columns: {sorted(missing)}"
        )

    if review["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in review")

    if metrics["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in metrics")

    # Normalize manually entered labels.
    for field in REVIEW_FIELDS:
        review[field] = (
            review[field]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        invalid = (
            set(review[field])
            - VALID_LABELS
            - {""}
        )

        if invalid:
            raise ValueError(
                f"Invalid labels in {field}: {invalid}"
            )

    reviewed_ids = set(review["event_id"])
    metric_ids = set(metrics["event_id"])

    if reviewed_ids != metric_ids:
        raise ValueError(
            "Review and metrics event IDs differ: "
            f"missing={sorted(metric_ids - reviewed_ids)}, "
            f"extra={sorted(reviewed_ids - metric_ids)}"
        )

    # Require every event to be assessed.
    incomplete = review[
        review[REVIEW_FIELDS].eq("").any(axis=1)
    ]

    if not incomplete.empty:
        raise ValueError(
            "Complete YES / NO / UNCERTAIN for events: "
            f"{incomplete['event_id'].tolist()}"
        )

    combined = metrics.merge(
        review[
            ["event_id", *REVIEW_FIELDS, "review_notes"]
        ],
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    return combined


# ==================================================
# SUMMARY
# ==================================================

def summarize_field(df, field):
    counts = df[field].value_counts()

    yes = int(counts.get("YES", 0))
    no = int(counts.get("NO", 0))
    uncertain = int(counts.get("UNCERTAIN", 0))

    assessed = yes + no

    return {
        "criterion": field,
        "yes": yes,
        "no": no,
        "uncertain": uncertain,
        "assessed": assessed,
        "total_events": len(df),
        "yes_rate_among_assessed": (
            yes / assessed if assessed else float("nan")
        ),
        "assessment_coverage": (
            assessed / len(df) if len(df) else float("nan")
        ),
    }


def main():
    data = load_data()

    summaries = pd.DataFrame([
        summarize_field(data, field)
        for field in REVIEW_FIELDS
    ])

    summaries.to_csv(OUTPUT_CSV, index=False)

    print("\n--- PHASE 7.5 MANUAL REVIEW ---")
    print("Episodes:", len(data))
    print("Unique track IDs:", data["track_id"].nunique())

    print("\nManual review results:")
    print(
        summaries[
            [
                "criterion",
                "yes",
                "no",
                "uncertain",
                "assessment_coverage",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nTemporal behavior:")
    print(
        "Mean confirmation delay:",
        round(
            data["confirmation_delay_frames"].mean(),
            2,
        ),
        "frames",
    )

    print(
        "Median confirmation delay:",
        round(
            data["confirmation_delay_frames"].median(),
            2,
        ),
        "frames",
    )

    print(
        "Maximum episode length:",
        int(data["episode_frames"].max()),
        "frames",
    )

    # ------------------------------------------
    # Markdown report
    # ------------------------------------------

    lines = [
        "# RoadVision — Phase 7.5 Evaluation",
        "",
        "## Event statistics",
        "",
        f"- Confirmed evidence episodes: {len(data)}",
        f"- Unique track IDs: {data['track_id'].nunique()}",
        (
            "- Mean confirmation delay: "
            f"{data['confirmation_delay_frames'].mean():.2f} frames"
        ),
        (
            "- Maximum episode length: "
            f"{int(data['episode_frames'].max())} frames"
        ),
        "",
        "## Manual review",
        "",
        "| Criterion | Yes | No | Uncertain | Assessed |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in summaries.itertuples(index=False):
        lines.append(
            f"| {row.criterion} | {row.yes} | {row.no} | "
            f"{row.uncertain} | {row.assessed} |"
        )

    lines += [
        "",
        "## Individual episodes",
        "",
        "| Event | Track | Frames | Approach | Lane | Tracking |",
        "|---|---:|---|---|---|---|",
    ]

    for row in data.itertuples(index=False):
        lines.append(
            f"| {row.event_id} | {row.track_id} | "
            f"{row.start_frame}–{row.end_frame} | "
            f"{row.apparent_approach} | "
            f"{row.ego_lane_correct} | "
            f"{row.tracking_stable} |"
        )

    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- Episodes indicate algorithmically stabilized "
        "image-space approach evidence.",
        "- Manual labels represent qualitative review, "
        "not instrumented ground truth.",
        "- UNCERTAIN judgments are reported explicitly "
        "and are not counted as successful events.",
        "- Review is limited to detected episodes; "
        "false negatives cannot be estimated.",
        "- Multiple episodes may correspond to the "
        "same physical vehicle.",
        "- Confirmation delay is relative to the "
        "algorithmic episode start, not the actual "
        "onset of a physical hazard.",
        "- No validated metric TTC or collision-risk "
        "accuracy is established.",
        "",
    ]

    OUTPUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
