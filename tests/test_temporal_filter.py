
from src.lanes.temporal_filter import TemporalLaneFilter


def make_lane(right_bottom_x):
    return {
        "valid": True,
        "left_fit": (570, 972, 810, 734),
        "right_fit": (right_bottom_x, 972, 1080, 734),
    }


# --------------------------------------------
# Test 1: Reject isolated jump
# --------------------------------------------

print("\n--- TEST 1: ISOLATED JUMP ---")

lane_filter = TemporalLaneFilter(
    max_jump=100,
    max_reference_age=5,
)

sequence = [
    (27, 1624),
    (28, 1291),
    (29, 1628),
]

expected = [
    (True, "initialized"),
    (False, "temporal_jump"),
    (True, "consistent"),
]

for (frame, right_x), expected_result in zip(
    sequence, expected
):
    result = lane_filter.update(
        frame,
        make_lane(right_x),
    )

    print(frame, right_x, result)
    assert result == expected_result


# --------------------------------------------
# Test 2: Recovery after stale reference
# --------------------------------------------

print("\n--- TEST 2: RECOVERY CONFIRMATION ---")

lane_filter.reset()

sequence = [
    (100, 1500),
    (101, 1200),
    (102, 1210),
    (103, 1220),
    (104, 1230),
    (105, 1240),
    (106, 1270),
    (107, 1280),
    (108, 1285),
]

for frame, right_x in sequence:
    accepted, reason = lane_filter.update(
        frame,
        make_lane(right_x),
    )

    print(
        f"Frame {frame}: "
        f"accepted={accepted}, "
        f"reason={reason}"
    )

assert lane_filter.last_accepted_frame == 108


# --------------------------------------------
# Test 3: Inconsistent recovery candidates
# --------------------------------------------

print("\n--- TEST 3: FAILED CONFIRMATION ---")

lane_filter.reset()

sequence = [
    (200, 1500),
    (206, 1200),
    (207, 1600),
    (208, 1590),
]

expected = [
    (True, "initialized"),
    (False, "pending_recovery"),
    (False, "pending_recovery"),
    (True, "recovered"),
]

for (frame, right_x), expected_result in zip(
    sequence, expected
):
    result = lane_filter.update(
        frame,
        make_lane(right_x),
    )

    print(frame, right_x, result)
    assert result == expected_result


print("\nAll temporal filter tests passed!")
