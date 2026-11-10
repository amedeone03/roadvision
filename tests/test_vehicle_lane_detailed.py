
from src.association.vehicle_lane import (
    build_lane_polygon,
    classify_vehicle_lane_detailed,
)


def test_detailed_vehicle_lane_association():

    left_fit = (517, 972, 779, 734)
    right_fit = (1540, 972, 1059, 734)

    polygon = build_lane_polygon(
        left_fit,
        right_fit,
    )

    test_cases = [
        (
            "Inside ego lane",
            (900, 700, 1100, 850),
            ("IN_LANE", "inside_polygon"),
        ),
        (
            "Outside ego lane",
            (200, 700, 400, 850),
            ("OUT_LANE", "outside_polygon"),
        ),
        (
            "Distant vehicle",
            (700, 500, 820, 630),
            ("UNKNOWN", "above_lane_coverage"),
        ),
    ]

    for name, bbox, expected in test_cases:

        result = classify_vehicle_lane_detailed(
            bbox,
            polygon,
        )

        print(f"{name}: {result}")

        assert result == expected

    result = classify_vehicle_lane_detailed(
        (900, 700, 1100, 850),
        None,
    )

    assert result == ("UNKNOWN", "lane_unavailable")
