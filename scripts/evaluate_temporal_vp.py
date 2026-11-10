
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "runs/motion/full_video_vp/full_video_vp_results.csv"
)

OUTPUT_PATH = Path(
    "runs/motion/full_video_vp/temporal_vp_results.csv"
)

METHODS = ["ORIGINAL", "INPAINTED", "EDGE_MASKED"]

# Initial experimental parameters, not validated limits.
ALPHA = 0.3
MAX_JUMP_PX = 25.0
CONFIRM_FRAMES = 3
CONFIRM_TOLERANCE_PX = 12.0
MAX_MISSING_FRAMES = 5


class TemporalVPFilter:
    def __init__(self):
        self.reference = None
        self.last_valid_frame = None
        self.pending = []
        self.last_pending_frame = None

    def update(self, frame, value):
        """
        Causal filtering of the VP vertical coordinate.

        Returns (filtered_y, reason).

        A rejected measurement produces NaN;
        previous values are not silently reused.
        """

        if pd.isna(value):
            self.pending.clear()
            self.last_pending_frame = None
            return np.nan, "missing"

        value = float(value)

        if not np.isfinite(value):
            return np.nan, "invalid"

        if self.reference is None:
            self.reference = value
            self.last_valid_frame = frame
            return value, "initialized"

        # Reset after a sufficiently long interval
        # without an accepted measurement.
        if frame - self.last_valid_frame > MAX_MISSING_FRAMES:
            self.reference = value
            self.last_valid_frame = frame
            self.pending.clear()
            self.last_pending_frame = None
            return value, "reinitialized"

        difference = abs(value - self.reference)

        if difference <= MAX_JUMP_PX:
            self.pending.clear()
            self.last_pending_frame = None

            self.reference = (
                ALPHA * value
                + (1 - ALPHA) * self.reference
            )

            self.last_valid_frame = frame

            return self.reference, "accepted"

        # Large change: require repeated evidence.
        if (
            self.last_pending_frame is None
            or frame != self.last_pending_frame + 1
        ):
            self.pending = [value]
        else:
            self.pending.append(value)

            if len(self.pending) > CONFIRM_FRAMES:
                self.pending.pop(0)

        self.last_pending_frame = frame

        if len(self.pending) == CONFIRM_FRAMES:
            spread = (
                max(self.pending) - min(self.pending)
            )

            if spread <= CONFIRM_TOLERANCE_PX:
                self.reference = float(
                    np.median(self.pending)
                )

                self.last_valid_frame = frame
                self.pending.clear()
                self.last_pending_frame = None

                return self.reference, "confirmed_shift"

        return np.nan, "jump_rejected"


def main():
    df = pd.read_csv(INPUT_PATH)

    required = {
        "frame", "method", "vp_y", "vp_valid"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.sort_values(
        ["method", "frame"]
    ).reset_index(drop=True)

    df["filtered_vp_y"] = np.nan
    df["filter_reason"] = ""

    for method in METHODS:
        filter_state = TemporalVPFilter()

        indices = df.index[
            df["method"] == method
        ]

        for idx in indices:
            row = df.loc[idx]

            raw_y = (
                row["vp_y"]
                if bool(row["vp_valid"])
                else np.nan
            )

            filtered, reason = filter_state.update(
                int(row["frame"]),
                raw_y,
            )

            df.at[idx, "filtered_vp_y"] = filtered
            df.at[idx, "filter_reason"] = reason

    # Temporal changes require consecutive
    # frames and valid values on both frames.
    df["previous_filtered_y"] = (
        df.groupby("method")[
            "filtered_vp_y"
        ].shift(1)
    )

    df["previous_frame"] = (
        df.groupby("method")["frame"].shift(1)
    )

    consecutive = (
        df["frame"] - df["previous_frame"] == 1
    )

    df["filtered_abs_delta_y"] = (
        df["filtered_vp_y"]
        - df["previous_filtered_y"]
    ).abs().where(consecutive)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- TEMPORAL VP EVALUATION ---")

    for method in METHODS:
        subset = df[df["method"] == method]

        raw_valid = int(
            subset["vp_y"].notna().sum()
        )

        filtered_valid = int(
            subset["filtered_vp_y"].notna().sum()
        )

        changes = subset[
            "filtered_abs_delta_y"
        ].dropna()

        print(f"\n{method}")
        print(f"Raw valid: {raw_valid}/{len(subset)}")
        print(
            f"Filtered valid: "
            f"{filtered_valid}/{len(subset)}"
        )

        print("Filter reasons:")
        print(
            subset["filter_reason"]
            .value_counts()
            .to_string()
        )

        if not changes.empty:
            print(
                "Median filtered jump: "
                f"{changes.median():.2f} px"
            )
            print(
                "P95 filtered jump: "
                f"{changes.quantile(0.95):.2f} px"
            )
            print(
                "Filtered jumps >20 px:",
                int((changes > 20).sum())
            )

        # Compare filtered values against the
        # existing manual diagnostic references.
        if "manual_y" in subset.columns:
            matched = subset[
                subset["manual_y"].notna()
                & subset["filtered_vp_y"].notna()
            ]

            print(
                "Matched manual references:",
                len(matched)
            )

            if not matched.empty:
                errors = (
                    matched["filtered_vp_y"]
                    - matched["manual_y"]
                ).abs()

                print(
                    "Median reference error: "
                    f"{errors.median():.2f} px"
                )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nTemporal VP experiment complete!")


if __name__ == "__main__":
    main()
