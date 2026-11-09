
class TemporalLaneFilter:
    """
    Temporal lane consistency filter with
    two-measurement recovery confirmation.

    Does not perform smoothing or prediction.
    """

    def __init__(
        self,
        max_jump=100,
        max_reference_age=5,
        max_candidate_gap=2,
    ):
        self.max_jump = max_jump
        self.max_reference_age = max_reference_age
        self.max_candidate_gap = max_candidate_gap

        self.reset()

    def reset(self):
        # Last trusted measurement
        self.last_accepted = None
        self.last_accepted_frame = None

        # Provisional recovery measurement
        self.pending_candidate = None
        self.pending_frame = None

        # Enforce chronological updates
        self.last_processed_frame = None

    def _clear_pending(self):
        self.pending_candidate = None
        self.pending_frame = None

    def _start_pending(self, current, frame_number):
        self.pending_candidate = current
        self.pending_frame = frame_number

    def _accept(self, current, frame_number):
        self.last_accepted = current
        self.last_accepted_frame = frame_number
        self._clear_pending()

    def _maximum_jump(self, current, reference):
        return max(
            abs(a - b)
            for a, b in zip(current, reference)
        )

    def update(self, frame_number, lane_info):
        """
        Returns:
            accepted: bool
            reason: str
        """

        # ------------------------------------------
        # 1. Check frame order
        # ------------------------------------------

        if (
            self.last_processed_frame is not None
            and frame_number <= self.last_processed_frame
        ):
            raise ValueError(
                "Frame numbers must increase monotonically"
            )

        self.last_processed_frame = frame_number

        # ------------------------------------------
        # 2. Geometric validation
        # ------------------------------------------

        if not lane_info["valid"]:
            self._clear_pending()
            return False, "invalid_geometry"

        left_fit = lane_info["left_fit"]
        right_fit = lane_info["right_fit"]

        current = (
            left_fit[0],
            left_fit[2],
            right_fit[0],
            right_fit[2],
        )

        # ------------------------------------------
        # 3. First initialization
        # ------------------------------------------

        if self.last_accepted is None:
            self._accept(current, frame_number)
            return True, "initialized"

        age = frame_number - self.last_accepted_frame

        # ------------------------------------------
        # 4. Recent trusted reference
        # ------------------------------------------

        if age <= self.max_reference_age:

            jump = self._maximum_jump(
                current,
                self.last_accepted,
            )

            if jump > self.max_jump:
                return False, "temporal_jump"

            self._accept(current, frame_number)
            return True, "consistent"

        # ------------------------------------------
        # 5. Reference is stale:
        #    begin or continue recovery
        # ------------------------------------------

        if self.pending_candidate is None:
            self._start_pending(current, frame_number)
            return False, "pending_recovery"

        candidate_age = (
            frame_number - self.pending_frame
        )

        # Require a nearby confirmation frame.
        if candidate_age > self.max_candidate_gap:
            self._start_pending(current, frame_number)
            return False, "pending_recovery"

        jump = self._maximum_jump(
            current,
            self.pending_candidate,
        )

        if jump > self.max_jump:
            # The provisional measurement was not
            # confirmed. Try the current one instead.
            self._start_pending(current, frame_number)
            return False, "pending_recovery"

        # ------------------------------------------
        # 6. Recovery confirmed
        # ------------------------------------------

        self._accept(current, frame_number)
        return True, "recovered"
