
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CSV_PATH = Path("runs/integrated_associations.csv")


# --------------------------------------------------
# 2. Load results
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Association CSV not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

if df.empty:
    raise RuntimeError("Association CSV is empty")

print("\n--- ROADVISION ASSOCIATION ANALYSIS ---")

print(f"Input: {CSV_PATH}")
print(f"Tracked observations: {len(df)}")
print(f"Frames with tracked observations: {df['frame'].nunique()}")
print(f"Unique tracking IDs: {df['track_id'].nunique()}")


# --------------------------------------------------
# 3. Inspect frame 2
# --------------------------------------------------

print("\n--- FRAME 2 ASSOCIATIONS ---")

frame_2 = df[df["frame"] == 2]

columns = [
    "track_id",
    "class_name",
    "contact_x",
    "contact_y",
    "lane_status",
    "lane_reason",
]

if frame_2.empty:
    print("No tracked observations in frame 2")
else:
    print(
        frame_2[columns].to_string(index=False)
    )


# --------------------------------------------------
# 4. Lane status distribution
# --------------------------------------------------

print("\n--- LANE ASSOCIATION COUNTS ---")

status_counts = df["lane_status"].value_counts()

print(status_counts)

print("\n--- LANE ASSOCIATION PERCENTAGES ---")

status_percentages = (
    df["lane_status"]
    .value_counts(normalize=True)
    * 100
)

print(status_percentages.round(2))


# --------------------------------------------------
# 5. UNKNOWN classifications
# --------------------------------------------------

print("\n--- UNKNOWN REASONS ---")

unknown = df[df["lane_status"] == "UNKNOWN"]

if unknown.empty:
    print("No UNKNOWN classifications")
else:
    print(
        unknown["lane_reason"].value_counts()
    )

    print("\nUNKNOWN percentages:")

    print(
        (
            unknown["lane_reason"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )


# --------------------------------------------------
# 6. Association by object class
# --------------------------------------------------

print("\n--- ASSOCIATIONS BY OBJECT CLASS ---")

class_table = pd.crosstab(
    df["class_name"],
    df["lane_status"],
)

print(class_table)


# --------------------------------------------------
# 7. Lane availability
# --------------------------------------------------

print("\n--- LANE AVAILABILITY ---")

# CSV contains one row per tracked observation,
# not necessarily one row per video frame.
observation_availability = (
    df["lane_valid"].astype(bool).mean()
)

print(
    "Lane availability among tracked observations: "
    f"{observation_availability:.2%}"
)

# Count unique frames with tracking observations
# and an accepted lane estimate.
frame_availability = (
    df.groupby("frame")["lane_valid"]
    .first()
    .astype(bool)
    .mean()
)

print(
    "Lane availability among frames with tracking: "
    f"{frame_availability:.2%}"
)


# --------------------------------------------------
# 8. Classification changes per track
# --------------------------------------------------

print("\n--- CLASSIFICATION CHANGES BY TRACK ---")

df = df.sort_values(
    ["track_id", "frame"]
).copy()

previous_status = (
    df.groupby("track_id")["lane_status"]
    .shift(1)
)

status_changed = (
    previous_status.notna()
    & df["lane_status"].ne(previous_status)
)

df["status_changed"] = status_changed

changes = (
    df.groupby("track_id")["status_changed"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 tracks by classification changes:")

print(changes.head(10))


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print("\n--- SUMMARY ---")

print(f"Total tracked observations: {len(df)}")
print(f"IN_LANE: {(df['lane_status'] == 'IN_LANE').sum()}")
print(f"OUT_LANE: {(df['lane_status'] == 'OUT_LANE').sum()}")
print(f"UNKNOWN: {(df['lane_status'] == 'UNKNOWN').sum()}")

print("\nAssociation analysis complete!")
