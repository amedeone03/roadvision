
import math

from src.lanes.vanishing_point import (
    estimate_vanishing_point,
    validate_vanishing_point,
)


def main():

    print("\n--- VANISHING POINT UNIT TESTS ---")

    # Both lines intersect at (960, 600).
    left = (660, 900, 860, 700)
    right = (1260, 900, 1060, 700)

    vp = estimate_vanishing_point(left, right)

    assert vp is not None
    assert math.isclose(vp[0], 960.0)
    assert math.isclose(vp[1], 600.0)

    print(f"Intersection: {vp}")

    valid, reason = validate_vanishing_point(
        vp, 1920, 1080
    )

    assert valid
    assert reason == "valid"

    # Parallel image lines have no finite VP.
    parallel_left = (500, 900, 600, 700)
    parallel_right = (900, 900, 1000, 700)

    assert estimate_vanishing_point(
        parallel_left, parallel_right
    ) is None

    assert estimate_vanishing_point(
        None, right
    ) is None

    print("All vanishing-point tests passed!")


if __name__ == "__main__":
    main()

def test_vanishing_point():
    main()
