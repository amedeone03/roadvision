
from src.lanes.temporal_filter import TemporalLaneFilter


def make_lane(right_bottom_x):
    return {
        "valid": True,
        "left_fit": (570, 972, 810, 734),
        "right_fit": (right_bottom_x, 972, 1080, 734),
    }


lane_filter = TemporalLaneFilter(
    max_jump=100,
    max_reference_age=5,
)

test_sequence = [
    (27, 1624),
    (28, 1291),
    (29, 1628),
]

for frame_number, right_x in test_sequence:

    accepted, reason = lane_filter.update(
        frame_number,
        make_lane(right_x),
    )

    print(
        f"Frame {frame_number}: "
        f"right_x={right_x}, "
        f"accepted={accepted}, "
        f"reason={reason}"
    )
