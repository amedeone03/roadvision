
import pandas as pd


df = pd.read_csv("runs/lane_temporal_results.csv")
df = df.sort_values("frame").reset_index(drop=True)

accepted = df["temporal_accepted"].astype(bool)
previous_accepted = accepted.shift(1, fill_value=False)

consecutive = (
    accepted
    & previous_accepted
    & df["frame"].diff().eq(1)
)

columns = [
    "left_bottom_x",
    "right_bottom_x",
    "left_top_x",
    "right_top_x",
]

print("\n--- FINAL ACCEPTED BOUNDARY STABILITY ---")

for column in columns:
    jumps = (df[column] - df[column].shift(1)).abs()
    jumps = jumps.where(consecutive)

    print(f"\n{column}")
    print(f"Comparable pairs: {jumps.notna().sum()}")
    print(f"Median jump: {jumps.median():.2f} px")
    print(f"95th percentile: {jumps.quantile(0.95):.2f} px")
    print(f"Maximum jump: {jumps.max():.2f} px")


print("\n--- REINITIALIZATION EVENTS ---")

reinitialized = df[
    df["temporal_reason"] == "reinitialized"
]

print(f"Reinitializations: {len(reinitialized)}")
print(
    reinitialized[
        ["frame", "timestamp", "right_bottom_x"]
    ].head(15).to_string(index=False)
)


print("\n--- LONGEST OUTPUT GAPS ---")

missing = ~accepted

new_run = (
    missing.ne(missing.shift(1))
    | df["frame"].diff().ne(1)
)

run_ids = new_run.cumsum()

gaps = (
    df.loc[missing]
    .groupby(run_ids[missing])
    .agg(
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        length=("frame", "count"),
    )
)

if not gaps.empty:
    print(
        gaps.sort_values(
            "length", ascending=False
        ).head(10).to_string(index=False)
    )

print("\nTemporal filter evaluation complete!")
