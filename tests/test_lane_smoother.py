
from src.lanes.lane_smoother import LaneSmoother


def make_lane(right_x):
    return {
        "left_fit": (500, 972, 800, 734),
        "right_fit": (right_x, 972, 1100, 734),
    }


def test_lane_smoother():

    smoother = LaneSmoother(alpha=0.3)

    sequence = [
        (1500, True, "initialized"),
        (1520, True, "consistent"),
        (1490, True, "consistent"),
        (1510, True, "consistent"),
    ]

    print("\n--- EMA SMOOTHING TEST ---")

    for right_x, accepted, reason in sequence:
        left_fit, right_fit = smoother.update(
            make_lane(right_x),
            accepted,
            reason,
        )

        print(
            f"Raw right X: {right_x} | "
            f"Smoothed right X: {right_fit[0]}"
        )

    print("\n--- RECOVERY RESET TEST ---")

    left_fit, right_fit = smoother.update(
        make_lane(1100),
        True,
        "recovered",
    )

    print(f"Recovered right X: {right_fit[0]}")

    assert right_fit[0] == 1100
