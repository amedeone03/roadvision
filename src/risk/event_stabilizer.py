
import math


class ApproachEventStabilizer:
    """
    Causal stabilizer for ego-lane approach evidence.

    CONFIRMED means sufficient recent evidence of
    apparent expansion in the ego lane.

    HELD means an earlier confirmed episode remains
    open, but the current frame is not confirmed.

    Neither state is a collision warning.
    """

    def __init__(
        self,
        min_confirmations=3,
        max_gap=2,
    ):
        if min_confirmations < 1 or max_gap < 0:
            raise ValueError("Invalid stabilizer parameters")

        self.min_confirmations = min_confirmations
        self.max_gap = max_gap
        self.completed_events = []

        self.previous_frame = None
        self._reset_candidate()

    def _reset_candidate(self):
        self.start_frame = None
        self.last_positive_frame = None
        self.confirmation_frame = None

        self.positive_count = 0
        self.interruption_count = 0
        self.total_interruptions = 0

        self.gap = 0
        self.confirmed = False
        self.min_tau = None

    def _close_candidate(self):
        if self.confirmed:
            self.completed_events.append({
                "start_frame": self.start_frame,
                "end_frame": self.last_positive_frame,
                "confirmation_frame": self.confirmation_frame,
                "duration_frames": (
                    self.last_positive_frame
                    - self.start_frame + 1
                ),
                "positive_frames": self.positive_count,
                "interruption_frames": (
                    self.last_positive_frame
                    - self.start_frame
                    + 1
                    - self.positive_count
                ),
                "min_tau_proxy_s": self.min_tau,
            })

        self._reset_candidate()

    def update(
        self,
        frame,
        motion_state,
        lane_status,
        tau_proxy=None,
    ):
        """
        Returns:
            NONE, PENDING, CONFIRMED, or HELD.

        Each instance should process one track only.
        """

        frame = int(frame)

        if (
            self.previous_frame is not None
            and frame <= self.previous_frame
        ):
            raise ValueError(
                "Frames must be strictly increasing"
            )

        # A missing track frame breaks continuity.
        if (
            self.previous_frame is not None
            and frame != self.previous_frame + 1
        ):
            self._close_candidate()

        self.previous_frame = frame

        # Explicit negative evidence.
        if (
            lane_status == "OUT_LANE"
            or motion_state == "RECEDING"
        ):
            self._close_candidate()
            return "NONE"

        positive = (
            lane_status == "IN_LANE"
            and motion_state == "APPROACHING"
        )

        if positive:
            # Start or continue an episode.
            if self.start_frame is None:
                self.start_frame = frame

            self.last_positive_frame = frame
            self.positive_count += 1
            self.gap = 0

            if tau_proxy is not None:
                tau = float(tau_proxy)

                if math.isfinite(tau) and tau > 0:
                    self.min_tau = (
                        tau
                        if self.min_tau is None
                        else min(self.min_tau, tau)
                    )

            if (
                not self.confirmed
                and self.positive_count
                >= self.min_confirmations
            ):
                self.confirmed = True
                self.confirmation_frame = frame

            return (
                "CONFIRMED"
                if self.confirmed
                else "PENDING"
            )

        # No current positive evidence.
        if self.start_frame is None:
            return "NONE"

        self.gap += 1
        self.total_interruptions += 1

        if self.gap > self.max_gap:
            self._close_candidate()
            return "NONE"

        return "HELD" if self.confirmed else "PENDING"

    def finish(self):
        """Finalize the last episode at end of input."""
        self._close_candidate()
        return list(self.completed_events)
