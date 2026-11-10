
from src.association.vehicle_lane import (
    classify_vehicle_lane_conservative,
    classify_vehicle_lane_detailed,
    build_lane_polygon,
)


LEFT = (517, 972, 779, 734)
RIGHT = (1540, 972, 1059, 734)


def check(name, bbox, expected_status, expected_reason):
    status, reason, normalized_margin = (
        classify_vehicle_lane_conservative(
            bbox, LEFT, RIGHT, margin=0.05
        )
    )

    print(
        f"{name}: {status}, {reason}, "
        f"margin={normalized_margin}"
    )

    assert status == expected_status
    assert reason == expected_reason


def main():

    print("\n--- CONSERVATIVE ASSOCIATION TESTS ---")

    # Vehicle clearly inside the ego lane.
    check(
        "Inside lane",
        (900, 700, 1100, 850),
        "IN_LANE",
        "inside_lateral_boundaries",
    )

    # Vehicle inside but close to the right boundary.
    check(
        "Near right boundary",
        (1260, 800, 1300, 850),
        "UNKNOWN",
        "near_lane_boundary",
    )

    # Vehicle clearly outside the right boundary.
    check(
        "Outside right boundary",
        (1450, 800, 1550, 850),
        "OUT_LANE",
        "outside_lateral_boundaries",
    )

    # Vehicle above the supported near-field interval.
    check(
        "Distant vehicle",
        (700, 500, 820, 630),
        "UNKNOWN",
        "above_lane_coverage",
    )

    # No lane geometry available.
    status, reason, value = (
        classify_vehicle_lane_conservative(
            (900, 700, 1100, 850),
            None,
            None,
        )
    )

    assert (status, reason, value) == (
        "UNKNOWN",
        "lane_unavailable",
        None,
    )

    # A zero margin should reproduce the original
    # geometric classification for valid lane fits.
    polygon = build_lane_polygon(LEFT, RIGHT)

    test_boxes = [
        (900, 700, 1100, 850),
        (1260, 800, 1300, 850),
        (1450, 800, 1550, 850),
        (700, 500, 820, 630),
    ]

    for bbox in test_boxes:

        baseline, _ = classify_vehicle_lane_detailed(
            bbox, polygon
        )

        new_status, _, _ = (
            classify_vehicle_lane_conservative(
                bbox, LEFT, RIGHT, margin=0.0
            )
        )

        assert baseline == new_status

    print("\nAll conservative association tests passed!")


if __name__ == "__main__":
    main()
