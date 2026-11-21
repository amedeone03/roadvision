
import argparse

import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Analyze temporal consistency of lane estimates"
)

parser.add_argument(
    "--csv",
    default="runs/lane_relaxed_results.csv",
)

parser.add_argument(
    "--threshold",
    type=float,
    default=100.0,
    help="Maximum diagnostic displacement in pixels",
)

args = parser.parse_args()


# --------------------------------------------------
# 2. Load estimates
# --------------------------------------------------

df = pd.read_csv(args.csv)

df = df.sort_values("frame").reset_index(drop=True)

if df.empty:
    raise RuntimeError("No lane measurements found")

valid = df["valid"].astype(bool)

# Only compare consecutive frames where both
# estimates passed geometric validation.
comparable = (
    valid
    & valid.shift(1, fill_value=False)
    & df["frame"].diff().eq(1)
)


# --------------------------------------------------
# 3. Calculate boundary displacements
# --------------------------------------------------

boundaries = [
    "left_bottom_x",
    "right_bottom_x",
    "left_top_x",
    "right_top_x",
]

for boundary in boundaries:

    jump_column = f"{boundary}_jump"

    df[jump_column] = (
        df[boundary] - df[boundary].shift(1)
    ).abs()

    df.loc[~comparable, jump_column] = float("nan")


# --------------------------------------------------
# 4. Flag suspicious measurements
# --------------------------------------------------

jump_columns = [
    f"{boundary}_jump"
    for boundary in boundaries
]

# A transition is suspicious if any boundary
# displacement exceeds the experimental threshold.
df["suspicious"] = (
    df[jump_columns] > args.threshold
).any(axis=1)

# Flagged transitions are only defined for
# eligible consecutive accepted frames.
df.loc[~comparable, "suspicious"] = False


# --------------------------------------------------
# 5. Results
# --------------------------------------------------

comparable_count = int(comparable.sum())
suspicious_count = int(df["suspicious"].sum())

print("\n--- TEMPORAL CONSISTENCY ANALYSIS ---")

print(f"Input: {args.csv}")
print(f"Jump threshold: {args.threshold:g} px")

print(f"Total frames: {len(df)}")
print(f"Geometrically valid frames: {valid.sum()}")
print(f"Comparable transitions: {comparable_count}")
print(f"Suspicious transitions: {suspicious_count}")

if comparable_count > 0:
    print(
        f"Suspicious transition rate: "
        f"{suspicious_count / comparable_count:.2%}"
    )

print("\n--- FLAGS BY BOUNDARY ---")

for boundary in boundaries:

    jump_column = f"{boundary}_jump"

    count = (
        df[jump_column] > args.threshold
    ).sum()

    print(f"{boundary}: {count}")


# --------------------------------------------------
# 6. Show suspicious frames
# --------------------------------------------------

suspicious_frames = df[df["suspicious"]].copy()

print("\n--- FIRST 20 SUSPICIOUS TRANSITIONS ---")

if suspicious_frames.empty:
    print("No suspicious transitions found")
else:
    print(
        suspicious_frames[
            [
                "frame",
                "timestamp",
                "left_bottom_x_jump",
                "right_bottom_x_jump",
                "left_top_x_jump",
                "right_top_x_jump",
            ]
        ].head(20).to_string(index=False)
    )


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

output_path = "runs/lane_temporal_analysis.csv"

df.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
print("Temporal analysis complete!")
