
VALID_LANES = {
    "IN_LANE",
    "OUT_LANE",
    "UNKNOWN",
}

VALID_MOTIONS = {
    "APPROACHING",
    "STABLE",
    "RECEDING",
    "UNKNOWN",
}


def classify_lane_evidence(
    motion_state,
    lane_status,
):
    """
    Combine apparent motion and lane association.

    Returns an evidence category, not a
    collision risk level.
    """

    if motion_state not in VALID_MOTIONS:
        raise ValueError(
            f"Unexpected motion state: {motion_state}"
        )

    if lane_status not in VALID_LANES:
        raise ValueError(
            f"Unexpected lane status: {lane_status}"
        )

    if motion_state == "UNKNOWN":
        return "MOTION_UNAVAILABLE"

    if motion_state != "APPROACHING":
        return "NO_APPROACH_EVIDENCE"

    if lane_status == "IN_LANE":
        return "EGO_LANE_APPROACH"

    if lane_status == "OUT_LANE":
        return "OTHER_LANE_APPROACH"

    return "LANE_UNCERTAIN_APPROACH"
