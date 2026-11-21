"""Causal per-track motion and approach-evidence state.

Each video run needs its own ``TrackMotionProcessor``. Track
histories do not survive ``finish()``, and two track IDs inside
one processor do not share samples or event state.

Growth uses ``time_seconds = (frame - 1) / fps`` and the
Theil-Sen slope from ``src/motion/robust_growth.py``. A frame
that is not the previous frame plus one starts a new segment,
which is the rule in ``scripts/evaluate_motion_smoothing.py``.
The slope is missing until the configured window is full.
Missing growth becomes approach ``UNKNOWN`` and a missing
inverse-expansion proxy.

``duration_s`` on a finished episode divides the frame count by
this processor's fps. The Phase 6 event CSV instead divided by
the constant 29.97. Classification does not use that constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roadvision.config import RoadVisionConfig
from src.motion.image_motion import bounding_box_features
from src.motion.robust_growth import theil_sen_slope
from src.risk.approach import classify_approach
from src.risk.event_stabilizer import ApproachEventStabilizer
from src.risk.lane_evidence import VALID_LANES, classify_lane_evidence
from src.risk.time_to_contact import estimate_tau_proxy


@dataclass(frozen=True, slots=True)
class MotionEstimate:
    """One observation after causal track processing."""

    frame: int
    track_id: int
    time_seconds: float
    height: float
    growth_robust_5: float | None
    approach_state: str
    tau_proxy_s: float | None
    evidence: str
    event_state: str


@dataclass(frozen=True, slots=True)
class ApproachEpisode:
    """A confirmed episode closed by a gap or by ``finish()``."""

    track_id: int
    start_frame: int
    end_frame: int
    confirmation_frame: int
    duration_frames: int
    duration_s: float
    positive_frames: int
    interruption_frames: int
    min_tau_proxy_s: float | None


class _TrackState:
    def __init__(self, stabilizer: ApproachEventStabilizer) -> None:
        self.stabilizer = stabilizer
        self.previous_frame: int | None = None
        self.times: list[float] = []
        self.heights: list[float] = []


class TrackMotionProcessor:
    """Online equivalent of the Phase 6 motion and evidence chain.

    The caller supplies each tracked box and its lane status.
    This class does not open videos or CSV files.
    """

    def __init__(
        self,
        fps: float,
        config: RoadVisionConfig | None = None,
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("fps must be a positive finite number")

        fps = float(fps)

        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be a positive finite number")

        self.fps = fps
        self.config = config if config is not None else RoadVisionConfig()
        window = self.config.motion.robust_window

        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError("robust_window must be an integer >= 2")

        self._window = window
        self._tracks: dict[int, _TrackState] = {}
        self._finished = False
        self._episodes: tuple[ApproachEpisode, ...] = ()

    def update(
        self,
        frame: int,
        track_id: int,
        bbox: tuple[float, float, float, float],
        lane_status: str,
    ) -> MotionEstimate:
        """Process one observation of one track.

        ``frame`` is the 1-based video frame index used by the
        experiment scripts. Samples for a track must arrive in
        increasing frame order.
        """

        if self._finished:
            raise RuntimeError(
                "This processor is finished. Create a new one for another video."
            )

        frame = _as_frame(frame, "frame")
        track_id = _as_frame(track_id, "track_id")

        if not isinstance(lane_status, str) or lane_status not in VALID_LANES:
            raise ValueError(
                "lane_status must be IN_LANE, OUT_LANE, or UNKNOWN"
            )

        height = bounding_box_features(bbox)["height"]
        state = self._tracks.get(track_id)

        if state is None:
            state = _TrackState(
                ApproachEventStabilizer(
                    min_confirmations=self.config.stabilizer.min_confirmations,
                    max_gap=self.config.stabilizer.max_gap,
                )
            )
            self._tracks[track_id] = state

        if (
            state.previous_frame is not None
            and frame <= state.previous_frame
        ):
            raise ValueError(
                f"Track {track_id} frames must be strictly increasing"
            )

        if (
            state.previous_frame is not None
            and frame != state.previous_frame + 1
        ):
            state.times.clear()
            state.heights.clear()

        time_seconds = (frame - 1) / self.fps
        state.times.append(time_seconds)
        state.heights.append(height)
        state.previous_frame = frame

        growth = self._growth(state)
        approach_state = classify_approach(
            growth,
            threshold=self.config.approach.growth_threshold,
        )
        tau_proxy = estimate_tau_proxy(
            growth,
            min_growth=self.config.time_to_contact.min_growth,
        )
        evidence = classify_lane_evidence(
            approach_state,
            lane_status,
        )
        event_state = state.stabilizer.update(
            frame,
            approach_state,
            lane_status,
            tau_proxy,
        )

        return MotionEstimate(
            frame=frame,
            track_id=track_id,
            time_seconds=time_seconds,
            height=height,
            growth_robust_5=growth,
            approach_state=approach_state,
            tau_proxy_s=tau_proxy,
            evidence=evidence,
            event_state=event_state,
        )

    def finish(self) -> tuple[ApproachEpisode, ...]:
        """Close every open confirmed episode.

        A later call returns the same episodes. Pending episodes
        that never reached confirmation are not emitted, matching
        ``ApproachEventStabilizer.finish``.
        """

        if self._finished:
            return self._episodes

        episodes: list[ApproachEpisode] = []

        for track_id, state in self._tracks.items():
            for event in state.stabilizer.finish():
                duration_frames = int(event["duration_frames"])
                min_tau = event["min_tau_proxy_s"]
                episodes.append(
                    ApproachEpisode(
                        track_id=int(track_id),
                        start_frame=int(event["start_frame"]),
                        end_frame=int(event["end_frame"]),
                        confirmation_frame=int(event["confirmation_frame"]),
                        duration_frames=duration_frames,
                        duration_s=duration_frames / self.fps,
                        positive_frames=int(event["positive_frames"]),
                        interruption_frames=int(event["interruption_frames"]),
                        min_tau_proxy_s=(
                            None if min_tau is None else float(min_tau)
                        ),
                    )
                )

        episodes.sort(key=lambda episode: (
            episode.start_frame,
            episode.track_id,
        ))
        self._episodes = tuple(episodes)
        self._finished = True
        return self._episodes

    def _growth(self, state: _TrackState) -> float | None:
        if len(state.heights) < self._window:
            return None

        times = state.times[-self._window:]
        heights = state.heights[-self._window:]
        return theil_sen_slope(times, np.log(heights))


def _as_frame(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")

    return int(value)
