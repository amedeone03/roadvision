
from pathlib import Path

import pandas as pd


INPUT = Path(
    "runs/motion/full_video_vp/temporal_vp_results.csv"
)

REFERENCE_FRAMES = [
    150, 250, 450, 650,
    850, 1023, 1200, 1450,
]

df = pd.read_csv(INPUT)

print("\n--- TEMPORAL VP REFERENCE CHECK ---")

reference = df[
    df["frame"].isin(REFERENCE_FRAMES)
].copy()

reference["filtered_error"] = (
    reference["filtered_vp_y"]
    - reference["manual_y"]
).abs()

print(
    reference[
        [
            "frame",
            "method",
            "manual_y",
            "vp_y",
            "filtered_vp_y",
            "filtered_error",
            "filter_reason",
        ]
    ]
    .sort_values(["frame", "method"])
    .round(2)
    .to_string(index=False)
)


print("\n--- ELEVATED VP REGION ---")

edge = df[
    df["method"] == "EDGE_MASKED"
].copy()

for start, end in [
    (630, 670),
    (830, 870),
]:

    segment = edge[
        edge["frame"].between(start, end)
    ]

    print(f"\nFrames {start}-{end}")

    print(
        "Raw valid:",
        segment["vp_y"].notna().sum()
    )

    print(
        "Filtered valid:",
        segment["filtered_vp_y"].notna().sum()
    )

    print(
        "Median raw VP:",
        round(segment["vp_y"].median(), 2)
    )

    print(
        "Median filtered VP:",
        round(
            segment["filtered_vp_y"].median(),
            2,
        )
    )

    print("\nFilter reasons:")
    print(
        segment["filter_reason"]
        .value_counts()
        .to_string()
    )

    print("\nSelected frames:")
    selected = segment[
        segment["frame"] % 5 == 0
    ]

    print(
        selected[
            [
                "frame",
                "vp_y",
                "filtered_vp_y",
                "filter_reason",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )
