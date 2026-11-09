
import numpy as np


class LaneSmoother:
    """
    Exponential moving average for accepted lane boundaries.

    Smooths all four endpoints:
        left bottom X
        left top X
        right bottom X
        right top X

    Does not predict missing measurements.
    """

    def __init__(self, alpha=0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = alpha
        self.smoothed = None

    def reset(self):
        """Remove the previous smoothed estimate."""
        self.smoothed = None

    def update(self, lane_info, accepted, reason=None):
        """
        Update the smoothing state only for accepted measurements.

        Returns:
            smoothed_left_fit, smoothed_right_fit
            or (None, None) when rejected.
        """

        if not accepted:
            return None, None

        left_fit = lane_info["left_fit"]
        right_fit = lane_info["right_fit"]

        current = np.array(
            [
                left_fit[0],
                left_fit[2],
                right_fit[0],
                right_fit[2],
            ],
            dtype=float,
        )

        # A newly recovered lane can have a very
        # different geometry from the previous one.
        # Do not blend across recovery events.
        if self.smoothed is None or reason in (
            "initialized",
            "recovered",
            "reinitialized",
        ):
            self.smoothed = current.copy()
        else:
            self.smoothed = (
                self.alpha * current
                + (1 - self.alpha) * self.smoothed
            )

        lb_x, lt_x, rb_x, rt_x = np.rint(
            self.smoothed
        ).astype(int)

        # Preserve vertical coordinates.
        left_smoothed = (
            int(lb_x),
            left_fit[1],
            int(lt_x),
            left_fit[3],
        )

        right_smoothed = (
            int(rb_x),
            right_fit[1],
            int(rt_x),
            right_fit[3],
        )

        return left_smoothed, right_smoothed
