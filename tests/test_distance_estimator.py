
import math

from src.motion.distance_estimator import (
    CameraModel,
    estimate_ground_distance,
    project_ground_distance,
)


def main():

    print("\n--- GROUND DISTANCE UNIT TESTS ---")

    camera = CameraModel(
        height_m=1.5,
        fy_px=1000.0,
        cy_px=540.0,
        pitch_deg=3.0,
    )

    print(f"Theoretical horizon: {camera.horizon_y:.2f} px")

    # ------------------------------------------
    # Test 1: Ground projection/inverse
    # ------------------------------------------

    for distance in [5, 10, 20, 50, 100]:

        image_y = project_ground_distance(
            distance,
            camera,
        )

        estimated = estimate_ground_distance(
            image_y,
            camera,
        )

        assert math.isclose(
            estimated,
            distance,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

        print(
            f"True: {distance:6.1f} m | "
            f"Image Y: {image_y:8.2f} px | "
            f"Estimated: {estimated:8.2f} m"
        )

    # ------------------------------------------
    # Test 2: Horizon and above-horizon points
    # ------------------------------------------

    assert estimate_ground_distance(
        camera.horizon_y - 10,
        camera,
    ) is None

    assert estimate_ground_distance(
        camera.horizon_y,
        camera,
    ) is None

    print("Horizon test passed")

    # ------------------------------------------
    # Test 3: Farther objects appear higher
    # ------------------------------------------

    y_10 = project_ground_distance(10, camera)
    y_50 = project_ground_distance(50, camera)

    assert y_50 < y_10

    print("Perspective ordering test passed")

    # ------------------------------------------
    # Test 4: Contact-point sensitivity
    # ------------------------------------------

    target_y = project_ground_distance(30, camera)

    distances = [
        estimate_ground_distance(
            target_y + error,
            camera,
        )
        for error in [-5, 0, 5]
    ]

    print("\nContact-point sensitivity at 30 m:")

    for error, distance in zip(
        [-5, 0, 5],
        distances,
    ):
        print(
            f"Y error: {error:+3d} px | "
            f"Distance: {distance:.2f} m"
        )

    assert distances[0] > distances[1] > distances[2]

    # ------------------------------------------
    # Test 5: Invalid camera parameters
    # ------------------------------------------

    try:
        CameraModel(
            height_m=-1,
            fy_px=1000,
            cy_px=540,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative camera height was accepted"
        )

    print("\nAll ground-distance tests passed!")


if __name__ == "__main__":
    main()

def test_ground_distance():
    main()