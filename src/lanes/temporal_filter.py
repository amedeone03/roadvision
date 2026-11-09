
class TemporalLaneFilter:
    """
    Experimental temporal consistency filter.

    Accepts geometrically valid lane estimates if their
    endpoints are sufficiently close to the last trusted
    estimate.

    Does not perform smoothing or prediction.
    """

    def __init__(
        self,
        max_jump=100,
        max_reference_age=5,
    ):
        self.max_jump = max_jump
        self.max_reference_age = max_reference_age

        self.last_accepted = None
        self.last_accepted_frame = None

    def reset(self):
        """Discard the previous trusted estimate."""

        self.last_accepted = None
        self.last_accepted_frame = None

    def update(self, frame_number, lane_info):
        """
        Process one lane estimate.

        Returns:
            accepted: bool
            reason: str
        """

        # --------------------------------------------------
        # 1. Reject geometrically invalid measurements
        # --------------------------------------------------

        if not lane_info["valid"]:
            return False, "invalid_geometry"

        left_fit = lane_info["left_fit"]
        right_fit = lane_info["right_fit"]

        current = (
            left_fit[0],
            left_fit[2],
            right_fit[0],
            right_fit[2],
        )

        # --------------------------------------------------
        # 2. Check whether the reference is recent
        # --------------------------------------------------

        if self.last_accepted is None:
            self.last_accepted = current
            self.last_accepted_frame = frame_number

            return True, "initialized"

        age = frame_number - self.last_accepted_frame

        if age <= 0:
            raise ValueError(
                "Frame numbers must increase monotonically"
            )

        if age > self.max_reference_age:
            self.last_accepted = current
            self.last_accepted_frame = frame_number

            return True, "reinitialized"

        # --------------------------------------------------
        # 3. Compare endpoint positions
        # --------------------------------------------------

        jumps = [
            abs(a - b)
            for a, b in zip(
                current,
                self.last_accepted,
            )
        ]

        maximum_jump = max(jumps)

        # --------------------------------------------------
        # 4. Accept or reject
        # --------------------------------------------------

        if maximum_jump > self.max_jump:
            return False, "temporal_jump"

        self.last_accepted = current
        self.last_accepted_frame = frame_number

        return True, "consistent"
