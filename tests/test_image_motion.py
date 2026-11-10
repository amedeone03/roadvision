
import math

from src.motion.image_motion import (
    bounding_box_features,
    calculate_image_motion,
)


def main():

    print("\n--- IMAGE MOTION UNIT TESTS ---")

    # A box that moves 10 px right and grows
    # from 100 px to 110 px high.
    previous = bounding_box_features(
        (100, 100, 200, 200)
    )

    current = bounding_box_features(
        (110, 95, 210, 205)
    )

    dt = 0.1

    motion = calculate_image_motion(
        previous,
        current,
        dt,
    )

    assert math.isclose(
        motion["velocity_x_px_s"],
        100.0,
    )

    assert math.isclose(
        motion["velocity_y_px_s"],
        0.0,
    )

    assert math.isclose(
        motion["height_growth_px_s"],
        100.0,
    )

    expected_log_growth = (
        math.log(110 / 100) / dt
    )

    assert math.isclose(
        motion["log_height_growth_s"],
        expected_log_growth,
    )

    assert math.isclose(
        previous["area"],
        10000,
    )

    # Reject invalid bounding-box geometry.
    try:
        bounding_box_features(
            (100, 100, 50, 200)
        )
        raise AssertionError(
            "Invalid box was accepted"
        )
    except ValueError:
        pass

    print("Horizontal velocity: 100 px/s")
    print("Vertical center velocity: 0 px/s")
    print("Height growth: 100 px/s")
    print(
        f"Log-height growth: "
        f"{expected_log_growth:.4f} 1/s"
    )

    print("\nAll image-motion tests passed!")


if __name__ == "__main__":
    main()

def test_image_motion():
    main()
