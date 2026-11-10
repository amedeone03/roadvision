
from src.association.vehicle_lane import (
    build_lane_polygon,
    get_bottom_center,
    is_vehicle_in_lane,
)


def test_vehicle_lane_association():

    left_fit = (517, 972, 779, 734)
    right_fit = (1540, 972, 1059, 734)

    polygon = build_lane_polygon(
        left_fit,
        right_fit,
    )

    test_vehicles = {
        "ego_lane": (900, 700, 1100, 850),
        "left_lane": (200, 700, 400, 850),
        "right_lane": (1450, 700, 1700, 850),
    }

    for vehicle_name, bbox in test_vehicles.items():

        point = get_bottom_center(bbox)

        inside = is_vehicle_in_lane(
            bbox,
            polygon,
        )

        print(
            f"{vehicle_name}: "
            f"bottom_center={point}, "
            f"inside_ego_lane={inside}"
        )

        if vehicle_name == "ego_lane":
            assert inside is True
        else:
            assert inside is False

    assert is_vehicle_in_lane(
        test_vehicles["ego_lane"],
        None,
    ) is None
