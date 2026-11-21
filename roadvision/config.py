"""Configuration for the RoadVision application.

Defaults are the values already used by the Phase 4 integrated
video script and the Phase 6 approach-evidence scripts. Each
leaf field records that origin in ``metadata["origin"]``.

Lane-detector numbers are copied here so they can be checked.
``detect_lane()`` does not accept them as arguments.
``load_config()`` raises ``ConfigError`` when they differ from
``src/lanes/lane_detector.py``.

A ``RoadVisionConfig`` stores no frame, track, or filter state.
Each video run must construct its own YOLO model,
``TemporalLaneFilter``, ``LaneSmoother``, per-track growth
window, and ``ApproachEventStabilizer``.

``UNKNOWN`` and ``HELD`` keep the meanings they have in ``src/``.
``UNKNOWN`` is an explicit refusal to classify a lane or a
motion. ``HELD`` means an earlier confirmed episode is still
open and the current frame is not new approach evidence.
``CONFIRMED`` is algorithmic evidence of apparent ego-lane
expansion, not a collision warning. Those labels are not
settings.

The observation schema omits the Phase 5 vanishing-point and
monocular-distance columns. Those experiments remain in
``scripts/``. ``motion_quality`` is one of the omitted columns:
the approach script required it during a CSV merge, and
``classify_approach`` does not read it.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


class ConfigError(ValueError):
    """Invalid RoadVision configuration or runtime path."""


SUPPORTED_DEVICES = ("cpu", "mps", "cuda")

# Labels produced by src/. They are not configuration knobs.
LANE_STATUSES = ("IN_LANE", "OUT_LANE", "UNKNOWN")
MOTION_STATES = ("APPROACHING", "STABLE", "RECEDING", "UNKNOWN")
EVENT_STATES = ("NONE", "PENDING", "CONFIRMED", "HELD")

# Constructed at the start of every video run, then discarded.
# ByteTrack state lives on the YOLO model when persist=True.
FRESH_STATE_PER_RUN = (
    "ultralytics.YOLO",
    "src.lanes.temporal_filter.TemporalLaneFilter",
    "src.lanes.lane_smoother.LaneSmoother",
    "per-track robust-growth window",
    "src.risk.event_stabilizer.ApproachEventStabilizer",
)

# scripts/run_integrated_video.py writes round(timestamp, 3).
# Scale-growth time is the unrounded (frame - 1) / measured_fps.
CSV_TIMESTAMP_DECIMALS = 3

# Episode duration_s in scripts/evaluate_event_stabilization.py
# divides frame counts by this constant. The test video's
# container rate is 30000/1001. Classification does not use
# this constant. Later reports compare duration_s explicitly.
PHASE6_DURATION_DIVISOR = 29.97

# Columns written by the application. approach_state is the
# 0.15/s baseline, stored in Phase 6 as approach_t015.
OBSERVATION_COLUMNS = (
    "frame",
    "timestamp",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "contact_x",
    "contact_y",
    "lane_status",
    "lane_reason",
    "baseline_status",
    "baseline_reason",
    "normalized_lateral_margin",
    "lane_valid",
    "lane_geometry_valid",
    "lane_temporal_reason",
    "height",
    "time_seconds",
    "growth_robust_5",
    "approach_state",
    "tau_proxy_s",
    "evidence",
    "event_state",
)

# Carried into runs/phase6/final/phase6_results.csv by the
# vanishing-point and monocular-distance scripts. They are not
# inputs to classify_approach, estimate_tau_proxy,
# classify_lane_evidence, or ApproachEventStabilizer.
EXCLUDED_PHASE5_FIELDS = (
    "filtered_vp_y",
    "distance_dynamic_m",
    "distance_vp_minus_10_m",
    "distance_vp_plus_10_m",
    "distance_sensitivity_range_m",
    "relative_distance_sensitivity",
    "high_distance_sensitivity",
    "previous_frame",
    "previous_contact_y",
    "previous_filtered_vp_y",
    "previous_distance_dynamic_m",
    "delta_distance_m",
    "contact_contribution_m",
    "vp_contribution_m",
    "vp_dominant",
    "apparent_relative_velocity_m_s",
    "scenario_complete_pair",
    "closing_speed_lr_5_m_s",
    "closing_speed_lr_9_m_s",
    "closing_speed_raw_m_s",
    "distance_uncertainty_complete",
    "distance_low_sensitivity",
    "point_low_sensitivity",
    "window_complete",
    "window_low_sensitivity",
    "window_max_sensitivity",
    "motion_quality",
    "closing_speed_qualified_m_s",
    "large_apparent_speed",
)

# Sensitivity passes in scripts/evaluate_approach.py.
# The running pipeline keeps the 0.15/s baseline only.
EXCLUDED_SENSITIVITY_FIELDS = (
    "approach_t010",
    "approach_t020",
)

# Earlier or diagnostic columns that the stabilizer replaced
# or that only describe frame-to-frame proxy jitter.
EXCLUDED_SUPERSEDED_FIELDS = (
    "approach_streak",
    "sustained_ego_approach",
    "abs_tau_change_s",
)

REPLAY_COLUMN_ALIASES = {
    "approach_state": "approach_t015",
}

# Later CSV replay: these strings must match exactly.
REPLAY_EXACT_CATEGORICAL_FIELDS = (
    "lane_status",
    "lane_reason",
    "baseline_status",
    "baseline_reason",
    "approach_state",
    "evidence",
    "event_state",
)

REPLAY_EXACT_EVENT_FIELDS = (
    "track_id",
    "start_frame",
    "end_frame",
    "confirmation_frame",
    "duration_frames",
    "positive_frames",
    "interruption_frames",
)

# Absolute tolerances for finite values recomputed from the
# stored, rounded boxes. Missing stays missing: None, blank,
# and NaN match each other and do not match a finite number.
# 1e-9 /s is far below the 0.15/s decision threshold.
REPLAY_ABSOLUTE_TOLERANCE = {
    "growth_robust_5": 1e-9,
    "tau_proxy_s": 1e-9,
    "min_tau_proxy_s": 1e-9,
    "height": 1e-9,
    "time_seconds": 1e-9,
    "normalized_lateral_margin": 1e-9,
    "confidence": 1e-6,
    "x1": 1e-6,
    "y1": 1e-6,
    "x2": 1e-6,
    "y2": 1e-6,
    "contact_x": 1e-6,
    "contact_y": 1e-6,
}

# Present in the Phase 6 event CSV, compared outside the
# tight table because the divisor differs from measured FPS.
REPLAY_FLOATS_WITH_KNOWN_DIFFERENCE = ("duration_s",)

# Contents of configs/bytetrack_threshold.yaml. The app points
# at that file instead of copying the numbers into settings
# that ByteTrack would not read.
SELECTED_TRACKER_FILE = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.10,
    "new_track_thresh": 0.40,
    "track_buffer": 30,
    "match_thresh": 0.80,
    "fuse_score": True,
}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LANE_DETECTOR_PATH = _REPO_ROOT / "src" / "lanes" / "lane_detector.py"

_LANE_ORIGIN = (
    "src/lanes/lane_detector.py. Module constants and the "
    "literals in detect_lane() and validate_lane_polygon(). "
    "detect_lane() takes no parameter override; a difference "
    "is a configuration error."
)


def _opt(default: Any, origin: str) -> Any:
    return field(default=default, metadata={"origin": origin})


@dataclass(frozen=True, slots=True)
class VideoConfig:
    input_path: str = _opt(
        "data/videos/test_drive.mp4",
        "scripts/run_integrated_video.py VIDEO_PATH",
    )
    output_dir: str = _opt(
        "runs/final",
        "Phase 8 CLI contract. The Phase 4 script wrote "
        "runs/integrated_final_output.mp4 and "
        "runs/integrated_final.csv instead.",
    )
    max_frames: int | None = _opt(
        None,
        "Phase 8 smoke-test switch. None processes every frame.",
    )


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    model_path: str = _opt(
        "yolo11n.pt",
        "scripts/run_integrated_video.py MODEL_PATH",
    )
    device: str = _opt(
        "mps",
        "scripts/run_integrated_video.py DEVICE. "
        "Availability is checked when a run starts, not here.",
    )
    confidence: float = _opt(
        0.25,
        "scripts/run_integrated_video.py CONFIDENCE. "
        "Separate from ByteTrack track_high_thresh, which is "
        "also 0.25 in configs/bytetrack_threshold.yaml.",
    )
    classes: tuple[int, ...] = _opt(
        (0, 1, 2, 3, 5, 7),
        "scripts/run_integrated_video.py ROAD_USER_CLASSES. "
        "COCO person, bicycle, car, motorcycle, bus, truck.",
    )


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    config_path: str = _opt(
        "configs/bytetrack_threshold.yaml",
        "scripts/run_integrated_video.py TRACKER_PATH. "
        "Selected Phase 2 baseline: new_track_thresh 0.40, "
        "track_buffer 30.",
    )


@dataclass(frozen=True, slots=True)
class LaneFilterConfig:
    max_jump: int = _opt(
        100,
        "scripts/run_integrated_video.py MAX_JUMP and "
        "TemporalLaneFilter.__init__ default.",
    )
    max_reference_age: int = _opt(
        5,
        "scripts/run_integrated_video.py MAX_REFERENCE_AGE and "
        "TemporalLaneFilter.__init__ default.",
    )
    max_candidate_gap: int = _opt(
        2,
        "scripts/run_integrated_video.py MAX_CANDIDATE_GAP and "
        "TemporalLaneFilter.__init__ default.",
    )


@dataclass(frozen=True, slots=True)
class LaneSmootherConfig:
    alpha: float = _opt(
        0.3,
        "scripts/run_integrated_video.py EMA_ALPHA and "
        "LaneSmoother.__init__ default.",
    )


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    lateral_margin: float = _opt(
        0.05,
        "scripts/run_integrated_video.py LATERAL_MARGIN and "
        "classify_vehicle_lane_conservative default. "
        "A contact point within this fraction of either "
        "boundary is UNKNOWN.",
    )


@dataclass(frozen=True, slots=True)
class MotionConfig:
    robust_window: int = _opt(
        5,
        "scripts/evaluate_robust_motion.py ROBUST_WINDOW and "
        "rolling_robust_growth default. The window uses the "
        "current frame and earlier frames of the same "
        "consecutive segment.",
    )


@dataclass(frozen=True, slots=True)
class ApproachConfig:
    growth_threshold: float = _opt(
        0.15,
        "src/risk/approach.py default and "
        "scripts/evaluate_approach.py THRESHOLDS['t015']. "
        "Units are 1/s of log bounding-box height. "
        "This is image-scale motion, not physical speed.",
    )


@dataclass(frozen=True, slots=True)
class TimeToContactConfig:
    min_growth: float = _opt(
        0.15,
        "src/risk/time_to_contact.py default and "
        "scripts/evaluate_time_to_contact.py THRESHOLD. "
        "The published pipeline uses the same 0.15 number as "
        "the approach baseline. The two settings are independent.",
    )


@dataclass(frozen=True, slots=True)
class StabilizerConfig:
    min_confirmations: int = _opt(
        3,
        "src/risk/event_stabilizer.py default and "
        "scripts/evaluate_event_stabilization.py "
        "MIN_CONFIRMATIONS.",
    )
    max_gap: int = _opt(
        2,
        "src/risk/event_stabilizer.py default and "
        "scripts/evaluate_event_stabilization.py MAX_GAP. "
        "Interruptions of this many frames can stay PENDING "
        "or HELD. OUT_LANE and RECEDING still close the episode.",
    )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    annotated_video: str = _opt(
        "annotated.mp4",
        "Phase 8 output contract, written inside video.output_dir.",
    )
    observations_csv: str = _opt(
        "observations.csv",
        "Phase 8 output contract, written inside video.output_dir.",
    )
    events_csv: str = _opt(
        "events.csv",
        "Phase 8 output contract, written inside video.output_dir.",
    )
    metadata_json: str = _opt(
        "metadata.json",
        "Phase 8 output contract, written inside video.output_dir.",
    )


@dataclass(frozen=True, slots=True)
class LaneDetectorConfig:
    blur_kernel: tuple[int, int] = _opt((5, 5), _LANE_ORIGIN)
    blur_sigma: float = _opt(0.0, _LANE_ORIGIN)
    canny_low: int = _opt(50, _LANE_ORIGIN)
    canny_high: int = _opt(150, _LANE_ORIGIN)
    min_abs_slope: float = _opt(0.40, _LANE_ORIGIN)
    max_abs_slope: float = _opt(3.0, _LANE_ORIGIN)
    hough_rho: float = _opt(1.0, _LANE_ORIGIN)
    hough_theta_deg: float = _opt(1.0, _LANE_ORIGIN)
    hough_threshold: int = _opt(40, _LANE_ORIGIN)
    min_line_length: int = _opt(40, _LANE_ORIGIN)
    max_line_gap: int = _opt(30, _LANE_ORIGIN)
    roi_bottom_left_x: float = _opt(0.20, _LANE_ORIGIN)
    roi_top_left_x: float = _opt(0.43, _LANE_ORIGIN)
    roi_top_y: float = _opt(0.52, _LANE_ORIGIN)
    roi_top_right_x: float = _opt(0.58, _LANE_ORIGIN)
    roi_bottom_right_x: float = _opt(0.95, _LANE_ORIGIN)
    fit_y_bottom: float = _opt(0.90, _LANE_ORIGIN)
    fit_y_top: float = _opt(0.68, _LANE_ORIGIN)
    reference_width_px: int = _opt(1920, _LANE_ORIGIN)
    min_bottom_width_px: int = _opt(650, _LANE_ORIGIN)
    max_bottom_width_px: int = _opt(1350, _LANE_ORIGIN)
    min_top_width_px: int = _opt(0, _LANE_ORIGIN)
    max_top_width_px: int = _opt(500, _LANE_ORIGIN)


@dataclass(frozen=True, slots=True)
class RoadVisionConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    lane_filter: LaneFilterConfig = field(default_factory=LaneFilterConfig)
    lane_smoother: LaneSmootherConfig = field(
        default_factory=LaneSmootherConfig
    )
    association: AssociationConfig = field(
        default_factory=AssociationConfig
    )
    motion: MotionConfig = field(default_factory=MotionConfig)
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    time_to_contact: TimeToContactConfig = field(
        default_factory=TimeToContactConfig
    )
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    lane_detector: LaneDetectorConfig = field(
        default_factory=LaneDetectorConfig
    )


def field_origins(
    cls: type = RoadVisionConfig,
    prefix: str = "",
) -> dict[str, str]:
    """Return ``section.field`` -> origin for every leaf setting."""

    found: dict[str, str] = {}
    hints = get_type_hints(cls)

    for item in fields(cls):
        path = f"{prefix}.{item.name}" if prefix else item.name
        child = hints[item.name]

        if isinstance(child, type) and is_dataclass(child):
            found.update(field_origins(child, path))
            continue

        found[path] = str(item.metadata.get("origin", ""))

    return found


def to_mapping(config: RoadVisionConfig | None = None) -> dict[str, Any]:
    """Return a YAML-friendly copy of a configuration."""

    if config is None:
        config = RoadVisionConfig()

    def convert(value: Any) -> Any:
        if is_dataclass(value):
            return {
                item.name: convert(getattr(value, item.name))
                for item in fields(value)
            }

        if isinstance(value, tuple):
            return [convert(item) for item in value]

        return value

    mapped = convert(config)

    if not isinstance(mapped, dict):
        raise ConfigError("Configuration mapping must be a dict")

    return mapped


def load_config(path: str | Path | None = None) -> RoadVisionConfig:
    """Load defaults, optionally merged with a YAML file.

    Paths in the file stay as written. They are resolved against
    the process working directory, as the experiment scripts do.
    A missing file is an error. An empty mapping uses the defaults.
    """

    data = to_mapping()

    if path is not None:
        _deep_update(data, _read_yaml(Path(path)), "")

    config = _build(data)
    _validate(config)
    return config


def validate_runtime_paths(config: RoadVisionConfig) -> None:
    """Check that the video, weights, and tracker file exist.

    This does not decode the video and does not create the
    output directory. Device availability is also left to the
    run, so a config can be loaded on another machine.
    """

    input_path = Path(config.video.input_path)

    if not input_path.is_file():
        raise ConfigError(f"Input video not found: {input_path}")

    model_path = Path(config.detector.model_path)

    if not model_path.is_file():
        raise ConfigError(f"YOLO weights not found: {model_path}")

    tracker_path = Path(config.tracker.config_path)

    if not tracker_path.is_file():
        raise ConfigError(
            f"Tracker configuration not found: {tracker_path}"
        )

    _validate_tracker_file(tracker_path)

    output_dir = Path(config.video.output_dir)

    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigError(
            f"Output path is not a directory: {output_dir}"
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if loaded is None:
        raise ConfigError(f"Configuration file is empty: {path}")

    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Configuration root must be a mapping: {path}"
        )

    return loaded


def _deep_update(
    base: dict[str, Any],
    override: dict[str, Any],
    prefix: str,
) -> None:
    if not isinstance(override, dict):
        location = prefix or "root"
        raise ConfigError(f"{location} must be a mapping")

    for key, value in override.items():
        if not isinstance(key, str):
            raise ConfigError(
                f"Configuration keys must be strings, got {key!r}"
            )

        path = f"{prefix}.{key}" if prefix else key

        if key not in base:
            raise ConfigError(f"Unknown configuration key: {path}")

        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{path} must be a mapping")

            _deep_update(base[key], value, path)
            continue

        if isinstance(value, dict):
            raise ConfigError(f"{path} must be a value, not a mapping")

        base[key] = value


def _build(data: dict[str, Any]) -> RoadVisionConfig:
    video = data["video"]
    detector = data["detector"]
    tracker = data["tracker"]
    lane_filter = data["lane_filter"]
    lane_smoother = data["lane_smoother"]
    association = data["association"]
    motion = data["motion"]
    approach = data["approach"]
    time_to_contact = data["time_to_contact"]
    stabilizer = data["stabilizer"]
    output = data["output"]
    lane = data["lane_detector"]

    return RoadVisionConfig(
        video=VideoConfig(
            input_path=_as_str(video["input_path"], "video.input_path"),
            output_dir=_as_str(video["output_dir"], "video.output_dir"),
            max_frames=_as_optional_int(
                video["max_frames"],
                "video.max_frames",
            ),
        ),
        detector=DetectorConfig(
            model_path=_as_str(
                detector["model_path"],
                "detector.model_path",
            ),
            device=_as_str(detector["device"], "detector.device"),
            confidence=_as_float(
                detector["confidence"],
                "detector.confidence",
            ),
            classes=_as_int_tuple(
                detector["classes"],
                "detector.classes",
            ),
        ),
        tracker=TrackerConfig(
            config_path=_as_str(
                tracker["config_path"],
                "tracker.config_path",
            ),
        ),
        lane_filter=LaneFilterConfig(
            max_jump=_as_int(
                lane_filter["max_jump"],
                "lane_filter.max_jump",
            ),
            max_reference_age=_as_int(
                lane_filter["max_reference_age"],
                "lane_filter.max_reference_age",
            ),
            max_candidate_gap=_as_int(
                lane_filter["max_candidate_gap"],
                "lane_filter.max_candidate_gap",
            ),
        ),
        lane_smoother=LaneSmootherConfig(
            alpha=_as_float(
                lane_smoother["alpha"],
                "lane_smoother.alpha",
            ),
        ),
        association=AssociationConfig(
            lateral_margin=_as_float(
                association["lateral_margin"],
                "association.lateral_margin",
            ),
        ),
        motion=MotionConfig(
            robust_window=_as_int(
                motion["robust_window"],
                "motion.robust_window",
            ),
        ),
        approach=ApproachConfig(
            growth_threshold=_as_float(
                approach["growth_threshold"],
                "approach.growth_threshold",
            ),
        ),
        time_to_contact=TimeToContactConfig(
            min_growth=_as_float(
                time_to_contact["min_growth"],
                "time_to_contact.min_growth",
            ),
        ),
        stabilizer=StabilizerConfig(
            min_confirmations=_as_int(
                stabilizer["min_confirmations"],
                "stabilizer.min_confirmations",
            ),
            max_gap=_as_int(
                stabilizer["max_gap"],
                "stabilizer.max_gap",
            ),
        ),
        output=OutputConfig(
            annotated_video=_as_str(
                output["annotated_video"],
                "output.annotated_video",
            ),
            observations_csv=_as_str(
                output["observations_csv"],
                "output.observations_csv",
            ),
            events_csv=_as_str(
                output["events_csv"],
                "output.events_csv",
            ),
            metadata_json=_as_str(
                output["metadata_json"],
                "output.metadata_json",
            ),
        ),
        lane_detector=LaneDetectorConfig(
            blur_kernel=_as_pair(
                lane["blur_kernel"],
                "lane_detector.blur_kernel",
            ),
            blur_sigma=_as_float(
                lane["blur_sigma"],
                "lane_detector.blur_sigma",
            ),
            canny_low=_as_int(
                lane["canny_low"],
                "lane_detector.canny_low",
            ),
            canny_high=_as_int(
                lane["canny_high"],
                "lane_detector.canny_high",
            ),
            min_abs_slope=_as_float(
                lane["min_abs_slope"],
                "lane_detector.min_abs_slope",
            ),
            max_abs_slope=_as_float(
                lane["max_abs_slope"],
                "lane_detector.max_abs_slope",
            ),
            hough_rho=_as_float(
                lane["hough_rho"],
                "lane_detector.hough_rho",
            ),
            hough_theta_deg=_as_float(
                lane["hough_theta_deg"],
                "lane_detector.hough_theta_deg",
            ),
            hough_threshold=_as_int(
                lane["hough_threshold"],
                "lane_detector.hough_threshold",
            ),
            min_line_length=_as_int(
                lane["min_line_length"],
                "lane_detector.min_line_length",
            ),
            max_line_gap=_as_int(
                lane["max_line_gap"],
                "lane_detector.max_line_gap",
            ),
            roi_bottom_left_x=_as_float(
                lane["roi_bottom_left_x"],
                "lane_detector.roi_bottom_left_x",
            ),
            roi_top_left_x=_as_float(
                lane["roi_top_left_x"],
                "lane_detector.roi_top_left_x",
            ),
            roi_top_y=_as_float(
                lane["roi_top_y"],
                "lane_detector.roi_top_y",
            ),
            roi_top_right_x=_as_float(
                lane["roi_top_right_x"],
                "lane_detector.roi_top_right_x",
            ),
            roi_bottom_right_x=_as_float(
                lane["roi_bottom_right_x"],
                "lane_detector.roi_bottom_right_x",
            ),
            fit_y_bottom=_as_float(
                lane["fit_y_bottom"],
                "lane_detector.fit_y_bottom",
            ),
            fit_y_top=_as_float(
                lane["fit_y_top"],
                "lane_detector.fit_y_top",
            ),
            reference_width_px=_as_int(
                lane["reference_width_px"],
                "lane_detector.reference_width_px",
            ),
            min_bottom_width_px=_as_int(
                lane["min_bottom_width_px"],
                "lane_detector.min_bottom_width_px",
            ),
            max_bottom_width_px=_as_int(
                lane["max_bottom_width_px"],
                "lane_detector.max_bottom_width_px",
            ),
            min_top_width_px=_as_int(
                lane["min_top_width_px"],
                "lane_detector.min_top_width_px",
            ),
            max_top_width_px=_as_int(
                lane["max_top_width_px"],
                "lane_detector.max_top_width_px",
            ),
        ),
    )


def _validate(config: RoadVisionConfig) -> None:
    _require_text(config.video.input_path, "video.input_path")
    _require_text(config.video.output_dir, "video.output_dir")

    if (
        config.video.max_frames is not None
        and config.video.max_frames < 1
    ):
        raise ConfigError("video.max_frames must be >= 1 or null")

    if config.detector.device not in SUPPORTED_DEVICES:
        allowed = ", ".join(SUPPORTED_DEVICES)
        raise ConfigError(
            "detector.device must be one of "
            f"{allowed}. Got {config.detector.device!r}. "
            "The value is not rewritten to another device."
        )

    if not math.isfinite(config.detector.confidence) or not (
        0.0 <= config.detector.confidence <= 1.0
    ):
        raise ConfigError(
            "detector.confidence must be finite and in [0, 1]"
        )

    if not config.detector.classes:
        raise ConfigError("detector.classes must not be empty")

    if any(class_id < 0 for class_id in config.detector.classes):
        raise ConfigError(
            "detector.classes must contain non-negative class ids"
        )

    _require_text(config.detector.model_path, "detector.model_path")
    _require_text(config.tracker.config_path, "tracker.config_path")

    if config.lane_filter.max_jump < 0:
        raise ConfigError("lane_filter.max_jump must be >= 0")

    if config.lane_filter.max_reference_age < 0:
        raise ConfigError("lane_filter.max_reference_age must be >= 0")

    if config.lane_filter.max_candidate_gap < 0:
        raise ConfigError("lane_filter.max_candidate_gap must be >= 0")

    # LaneSmoother rejects values outside (0, 1].
    if not math.isfinite(config.lane_smoother.alpha) or not (
        0.0 < config.lane_smoother.alpha <= 1.0
    ):
        raise ConfigError(
            "lane_smoother.alpha must be finite and in (0, 1]"
        )

    # classify_vehicle_lane_conservative rejects this range.
    if not math.isfinite(config.association.lateral_margin) or not (
        0.0 <= config.association.lateral_margin < 0.5
    ):
        raise ConfigError(
            "association.lateral_margin must be finite and in [0, 0.5)"
        )

    # rolling_robust_growth rejects windows below 2.
    if config.motion.robust_window < 2:
        raise ConfigError("motion.robust_window must be an integer >= 2")

    # classify_approach rejects non-positive thresholds.
    if (
        not math.isfinite(config.approach.growth_threshold)
        or config.approach.growth_threshold <= 0.0
    ):
        raise ConfigError(
            "approach.growth_threshold must be finite and > 0"
        )

    # estimate_tau_proxy rejects non-positive minimum growth.
    if (
        not math.isfinite(config.time_to_contact.min_growth)
        or config.time_to_contact.min_growth <= 0.0
    ):
        raise ConfigError(
            "time_to_contact.min_growth must be finite and > 0"
        )

    # ApproachEventStabilizer rejects these values.
    if config.stabilizer.min_confirmations < 1:
        raise ConfigError(
            "stabilizer.min_confirmations must be an integer >= 1"
        )

    if config.stabilizer.max_gap < 0:
        raise ConfigError("stabilizer.max_gap must be an integer >= 0")

    filenames = (
        config.output.annotated_video,
        config.output.observations_csv,
        config.output.events_csv,
        config.output.metadata_json,
    )
    labels = (
        "output.annotated_video",
        "output.observations_csv",
        "output.events_csv",
        "output.metadata_json",
    )

    for name, label in zip(filenames, labels):
        _require_filename(name, label)

    if len(set(filenames)) != len(filenames):
        raise ConfigError("Output file names must be distinct")

    _assert_lane_detector_locked(config.lane_detector)


def _assert_lane_detector_locked(cfg: LaneDetectorConfig) -> None:
    expected = LaneDetectorConfig()
    differences = [
        item.name
        for item in fields(LaneDetectorConfig)
        if getattr(cfg, item.name) != getattr(expected, item.name)
    ]

    if differences:
        details = ", ".join(
            (
                f"{name}={getattr(cfg, name)!r} "
                f"(locked {getattr(expected, name)!r})"
            )
            for name in differences
        )
        raise ConfigError(
            "lane_detector cannot override detect_lane(). "
            f"Differing fields: {details}."
        )

    literals = _assigned_literals(
        _LANE_DETECTOR_PATH,
        {
            "BLUR_KERNEL",
            "CANNY_LOW",
            "CANNY_HIGH",
            "MIN_ABS_SLOPE",
            "MAX_ABS_SLOPE",
            "HOUGH_THRESHOLD",
            "MIN_LINE_LENGTH",
            "MAX_LINE_GAP",
        },
    )
    expected_literals = {
        "BLUR_KERNEL": cfg.blur_kernel,
        "CANNY_LOW": cfg.canny_low,
        "CANNY_HIGH": cfg.canny_high,
        "MIN_ABS_SLOPE": cfg.min_abs_slope,
        "MAX_ABS_SLOPE": cfg.max_abs_slope,
        "HOUGH_THRESHOLD": cfg.hough_threshold,
        "MIN_LINE_LENGTH": cfg.min_line_length,
        "MAX_LINE_GAP": cfg.max_line_gap,
    }

    for name, value in expected_literals.items():
        if literals[name] != value:
            raise ConfigError(
                f"src/lanes/lane_detector.py {name} is "
                f"{literals[name]!r}, locked configuration is {value!r}"
            )

    source = _LANE_DETECTOR_PATH.read_text(encoding="utf-8")
    markers = (
        "BLUR_KERNEL, 0",
        "rho=1",
        "theta=np.pi / 180",
        f"int({cfg.roi_bottom_left_x:.2f} * width)",
        f"int({cfg.roi_top_left_x:.2f} * width)",
        f"int({cfg.roi_top_y:.2f} * height)",
        f"int({cfg.roi_top_right_x:.2f} * width)",
        f"int({cfg.roi_bottom_right_x:.2f} * width)",
        f"int({cfg.fit_y_bottom:.2f} * height)",
        f"int({cfg.fit_y_top:.2f} * height)",
        f"image_width / {cfg.reference_width_px}.0",
        f"{cfg.min_bottom_width_px} * scale",
        f"{cfg.max_bottom_width_px} * scale",
        f"MIN_TOP_WIDTH = {cfg.min_top_width_px}",
        f"{cfg.max_top_width_px} * scale",
    )
    missing = [marker for marker in markers if marker not in source]

    if missing:
        raise ConfigError(
            "src/lanes/lane_detector.py no longer contains the "
            f"locked literals: {missing}"
        )

    if cfg.blur_sigma != 0.0:
        raise ConfigError(
            "lane_detector.blur_sigma is locked to 0, the sigma "
            "argument of cv2.GaussianBlur in detect_lane()"
        )


def _assigned_literals(
    path: Path,
    names: set[str],
) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise ConfigError(f"Cannot parse {path}: {exc}") from exc

    found: dict[str, Any] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue

        target = node.targets[0]

        if not isinstance(target, ast.Name) or target.id not in names:
            continue

        try:
            found[target.id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError) as exc:
            raise ConfigError(
                f"{path.name}:{target.id} is not a literal, so the "
                "lane-detector lock cannot check it"
            ) from exc

    missing = names - set(found)

    if missing:
        raise ConfigError(
            "Missing lane-detector constants: "
            + ", ".join(sorted(missing))
        )

    return found


def _validate_tracker_file(path: Path) -> None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid tracker YAML in {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"Tracker YAML must be a mapping: {path}")

    if "tracker_type" not in loaded:
        raise ConfigError(
            f"Tracker YAML is missing tracker_type: {path}"
        )


def _as_str(value: Any, path: str) -> str:
    if isinstance(value, str):
        return value

    raise ConfigError(f"{path} must be a string")


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")

    value = float(value)

    if not math.isfinite(value):
        raise ConfigError(f"{path} must be finite")

    return value


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be an integer")

    if isinstance(value, float):
        if not value.is_integer():
            raise ConfigError(f"{path} must be an integer")

        value = int(value)

    return value


def _as_optional_int(value: Any, path: str) -> int | None:
    if value is None:
        return None

    return _as_int(value, path)


def _as_int_tuple(value: Any, path: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path} must be a list of integers")

    return tuple(_as_int(item, path) for item in value)


def _as_pair(value: Any, path: str) -> tuple[int, int]:
    parsed = _as_int_tuple(value, path)

    if len(parsed) != 2:
        raise ConfigError(f"{path} must contain two integers")

    return parsed


def _require_text(value: str, path: str) -> None:
    if not value.strip():
        raise ConfigError(f"{path} must not be empty")


def _require_filename(value: str, path: str) -> None:
    _require_text(value, path)
    candidate = Path(value)

    if candidate.name != value or value in {".", ".."}:
        raise ConfigError(
            f"{path} must be a file name inside the output directory"
        )
