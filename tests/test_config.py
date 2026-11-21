"""Configuration loading, defaults, and validation."""

import csv
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from roadvision.config import (
    CSV_TIMESTAMP_DECIMALS,
    EVENT_STATES,
    EXCLUDED_PHASE5_FIELDS,
    EXCLUDED_SENSITIVITY_FIELDS,
    EXCLUDED_SUPERSEDED_FIELDS,
    FRESH_STATE_PER_RUN,
    LANE_STATUSES,
    MOTION_STATES,
    OBSERVATION_COLUMNS,
    PHASE6_DURATION_DIVISOR,
    REPLAY_ABSOLUTE_TOLERANCE,
    REPLAY_COLUMN_ALIASES,
    REPLAY_EXACT_CATEGORICAL_FIELDS,
    REPLAY_EXACT_EVENT_FIELDS,
    REPLAY_FLOATS_WITH_KNOWN_DIFFERENCE,
    SELECTED_TRACKER_FILE,
    SUPPORTED_DEVICES,
    ConfigError,
    RoadVisionConfig,
    field_origins,
    load_config,
    validate_runtime_paths,
)
from src.risk.lane_evidence import VALID_LANES, VALID_MOTIONS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / "configs" / "roadvision.yaml"
PHASE6_CSV = ROOT / "runs" / "phase6" / "final" / "phase6_results.csv"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "roadvision.yaml"
    path.write_text(
        textwrap.dedent(text).lstrip("\n"),
        encoding="utf-8",
    )
    return path


def test_builtin_defaults_match_checked_in_yaml():
    assert load_config() == load_config(DEFAULT_YAML)


def test_published_numeric_defaults():
    config = load_config()

    assert config.video.input_path == "data/videos/test_drive.mp4"
    assert config.video.output_dir == "runs/final"
    assert config.video.max_frames is None

    assert config.detector.model_path == "yolo11n.pt"
    assert config.detector.device == "mps"
    assert config.detector.confidence == 0.25
    assert config.detector.classes == (0, 1, 2, 3, 5, 7)

    assert config.tracker.config_path == "configs/bytetrack_threshold.yaml"
    assert config.lane_filter.max_jump == 100
    assert config.lane_filter.max_reference_age == 5
    assert config.lane_filter.max_candidate_gap == 2
    assert config.lane_smoother.alpha == 0.3
    assert config.association.lateral_margin == 0.05
    assert config.motion.robust_window == 5
    assert config.approach.growth_threshold == 0.15
    assert config.time_to_contact.min_growth == 0.15
    assert config.stabilizer.min_confirmations == 3
    assert config.stabilizer.max_gap == 2
    assert CSV_TIMESTAMP_DECIMALS == 3
    assert PHASE6_DURATION_DIVISOR == 29.97


def test_selected_tracker_file_is_unchanged():
    path = ROOT / RoadVisionConfig().tracker.config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded == SELECTED_TRACKER_FILE


def test_every_setting_records_an_origin():
    origins = field_origins()

    assert "approach.growth_threshold" in origins
    assert "lane_detector.canny_low" in origins

    for path, origin in origins.items():
        assert origin.strip(), path
        assert (
            "scripts/" in origin
            or "src/" in origin
            or "Phase 8" in origin
        ), path


def test_status_labels_match_source_vocabularies():
    assert set(LANE_STATUSES) == VALID_LANES
    assert set(MOTION_STATES) == VALID_MOTIONS
    assert EVENT_STATES == ("NONE", "PENDING", "CONFIRMED", "HELD")
    assert "HELD" not in LANE_STATUSES
    assert "UNKNOWN" in LANE_STATUSES
    assert "UNKNOWN" in MOTION_STATES
    assert "CONFIRMED" in EVENT_STATES


def test_config_holds_no_temporal_state():
    config = load_config()

    assert FRESH_STATE_PER_RUN == (
        "ultralytics.YOLO",
        "src.lanes.temporal_filter.TemporalLaneFilter",
        "src.lanes.lane_smoother.LaneSmoother",
        "per-track robust-growth window",
        "src.risk.event_stabilizer.ApproachEventStabilizer",
    )
    assert not hasattr(config, "tracks")
    assert not hasattr(config, "lane_filter_state")

    with pytest.raises(FrozenInstanceError):
        config.approach.growth_threshold = 0.2  # type: ignore[misc]


def test_repeated_loads_are_independent():
    first = load_config()
    second = load_config()

    assert first == second
    assert first is not second
    assert first.detector.classes == second.detector.classes
    assert isinstance(first.detector.classes, tuple)


def test_empty_mapping_uses_defaults(tmp_path: Path):
    path = _write(tmp_path, "{}\n")

    assert load_config(path) == load_config()


def test_partial_override_preserves_other_defaults(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        video:
          max_frames: 12
        detector:
          device: cpu
        """,
    )
    config = load_config(path)

    assert config.video.max_frames == 12
    assert config.video.input_path == "data/videos/test_drive.mp4"
    assert config.detector.device == "cpu"
    assert config.detector.confidence == 0.25
    assert config.approach.growth_threshold == 0.15
    assert config.stabilizer.max_gap == 2


def test_approach_threshold_and_tau_floor_are_independent(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        approach:
          growth_threshold: 0.2
        """,
    )
    config = load_config(path)

    assert config.approach.growth_threshold == 0.2
    assert config.time_to_contact.min_growth == 0.15


def test_lane_detector_override_is_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        lane_detector:
          canny_low: 40
        """,
    )

    with pytest.raises(ConfigError, match="cannot override detect_lane"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        approach:
          deadband: 0.15
        """,
    )

    with pytest.raises(ConfigError, match="Unknown configuration key"):
        load_config(path)


def test_missing_config_file_is_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_empty_file_is_rejected(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_non_mapping_root_is_rejected(tmp_path: Path):
    path = _write(tmp_path, "- just-a-list\n")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_invalid_yaml_is_rejected(tmp_path: Path):
    path = tmp_path / "broken.yaml"
    path.write_text("approach: [\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("approach:\n  growth_threshold: 0\n", "growth_threshold"),
        ("approach:\n  growth_threshold: .nan\n", "growth_threshold"),
        ("time_to_contact:\n  min_growth: 0\n", "min_growth"),
        ("association:\n  lateral_margin: 0.5\n", "lateral_margin"),
        ("association:\n  lateral_margin: -0.01\n", "lateral_margin"),
        ("lane_smoother:\n  alpha: 0\n", "alpha"),
        ("lane_smoother:\n  alpha: 1.1\n", "alpha"),
        ("motion:\n  robust_window: 1\n", "robust_window"),
        ("stabilizer:\n  min_confirmations: 0\n", "min_confirmations"),
        ("stabilizer:\n  min_confirmations: true\n", "min_confirmations"),
        ("stabilizer:\n  max_gap: -1\n", "max_gap"),
        ("detector:\n  device: MPS\n", "device"),
        ("detector:\n  device: metal\n", "device"),
        ("detector:\n  confidence: -0.1\n", "confidence"),
        ("detector:\n  confidence: 1.1\n", "confidence"),
        ("detector:\n  classes: []\n", "classes"),
        ("detector:\n  classes: [-1]\n", "classes"),
        ("video:\n  max_frames: 0\n", "max_frames"),
        ("video:\n  max_frames: 1.5\n", "max_frames"),
        ("video:\n  input_path: ' '\n", "input_path"),
        ("lane_filter:\n  max_jump: -1\n", "max_jump"),
        ("lane_filter:\n  max_reference_age: -1\n", "max_reference_age"),
        (
            "output:\n  annotated_video: nested/out.mp4\n",
            "annotated_video",
        ),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, text: str, match: str):
    path = _write(tmp_path, text)

    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_constructor_boundaries_that_source_accepts(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        detector:
          confidence: 0
          device: cuda
        lane_smoother:
          alpha: 1
        association:
          lateral_margin: 0
        motion:
          robust_window: 2
        stabilizer:
          max_gap: 0
        video:
          max_frames: 1
        """,
    )
    config = load_config(path)

    assert config.detector.confidence == 0.0
    assert config.detector.device == "cuda"
    assert config.detector.device in SUPPORTED_DEVICES
    assert config.lane_smoother.alpha == 1.0
    assert config.association.lateral_margin == 0.0
    assert config.motion.robust_window == 2
    assert config.stabilizer.max_gap == 0
    assert config.video.max_frames == 1


def test_config_module_does_not_import_inference_libraries():
    source = Path(load_config.__code__.co_filename).read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "import cv2" not in source
    assert "import ultralytics" not in source
    assert "from ultralytics" not in source


def test_phase5_fields_stay_out_of_the_observation_schema():
    excluded = set(EXCLUDED_PHASE5_FIELDS)

    assert "motion_quality" in excluded
    assert "filtered_vp_y" in excluded
    assert "distance_dynamic_m" in excluded
    assert "closing_speed_qualified_m_s" in excluded
    assert excluded.isdisjoint(OBSERVATION_COLUMNS)
    assert set(EXCLUDED_SENSITIVITY_FIELDS).isdisjoint(OBSERVATION_COLUMNS)
    assert set(EXCLUDED_SUPERSEDED_FIELDS).isdisjoint(OBSERVATION_COLUMNS)
    assert excluded.isdisjoint(EXCLUDED_SENSITIVITY_FIELDS)
    assert "growth_robust_5" in OBSERVATION_COLUMNS
    assert "approach_state" in OBSERVATION_COLUMNS
    assert "approach_t015" not in OBSERVATION_COLUMNS
    assert len(OBSERVATION_COLUMNS) == len(set(OBSERVATION_COLUMNS))


def test_published_phase6_csv_contains_the_excluded_fields():
    if not PHASE6_CSV.is_file():
        pytest.skip("Phase 6 CSV is not in this checkout")

    with PHASE6_CSV.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    columns = set(header)

    assert set(EXCLUDED_PHASE5_FIELDS) <= columns
    assert set(EXCLUDED_SENSITIVITY_FIELDS) <= columns
    assert set(EXCLUDED_SUPERSEDED_FIELDS) <= columns
    assert "approach_t015" in columns
    assert "event_state" in columns


def test_replay_contract_separates_exact_and_numeric_fields():
    exact = set(REPLAY_EXACT_CATEGORICAL_FIELDS) | set(
        REPLAY_EXACT_EVENT_FIELDS
    )

    assert exact.isdisjoint(REPLAY_ABSOLUTE_TOLERANCE)
    assert "duration_s" in REPLAY_FLOATS_WITH_KNOWN_DIFFERENCE
    assert "duration_s" not in REPLAY_ABSOLUTE_TOLERANCE
    assert REPLAY_COLUMN_ALIASES["approach_state"] == "approach_t015"
    assert "event_state" in REPLAY_EXACT_CATEGORICAL_FIELDS
    assert "lane_status" in REPLAY_EXACT_CATEGORICAL_FIELDS

    for tolerance in REPLAY_ABSOLUTE_TOLERANCE.values():
        assert tolerance > 0.0
        assert tolerance <= 1e-6

    assert REPLAY_ABSOLUTE_TOLERANCE["growth_robust_5"] == 1e-9
    assert REPLAY_ABSOLUTE_TOLERANCE["tau_proxy_s"] == 1e-9


def test_runtime_paths_accept_a_stand_in_video(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    weights = tmp_path / "yolo11n.pt"
    tracker = tmp_path / "tracker.yaml"
    video.write_bytes(b"")
    weights.write_bytes(b"")
    tracker.write_text("tracker_type: bytetrack\n", encoding="utf-8")

    config_path = _write(
        tmp_path,
        f"""
        video:
          input_path: "{video}"
          output_dir: "{tmp_path / "out"}"
        detector:
          model_path: "{weights}"
        tracker:
          config_path: "{tracker}"
        """,
    )
    validate_runtime_paths(load_config(config_path))


def test_checked_in_runtime_files_when_present():
    config = load_config()

    if not Path(config.video.input_path).is_file():
        pytest.skip("Sample video is not in this checkout")

    if not Path(config.detector.model_path).is_file():
        pytest.skip("YOLO weights are not in this checkout")

    validate_runtime_paths(config)


def test_runtime_paths_report_missing_inputs(tmp_path: Path):
    config_path = _write(
        tmp_path,
        f"""
        video:
          input_path: "{tmp_path / "missing.mp4"}"
        """,
    )

    with pytest.raises(ConfigError, match="Input video not found"):
        validate_runtime_paths(load_config(config_path))


def test_runtime_paths_reject_a_file_used_as_output_dir(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    weights = tmp_path / "yolo11n.pt"
    tracker = tmp_path / "tracker.yaml"
    output = tmp_path / "not-a-directory"
    video.write_bytes(b"")
    weights.write_bytes(b"")
    tracker.write_text("tracker_type: bytetrack\n", encoding="utf-8")
    output.write_text("file", encoding="utf-8")

    config_path = _write(
        tmp_path,
        f"""
        video:
          input_path: "{video}"
          output_dir: "{output}"
        detector:
          model_path: "{weights}"
        tracker:
          config_path: "{tracker}"
        """,
    )

    with pytest.raises(ConfigError, match="not a directory"):
        validate_runtime_paths(load_config(config_path))


def test_runtime_paths_reject_a_tracker_file_without_type(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    weights = tmp_path / "yolo11n.pt"
    tracker = tmp_path / "tracker.yaml"
    video.write_bytes(b"")
    weights.write_bytes(b"")
    tracker.write_text("track_buffer: 30\n", encoding="utf-8")

    config_path = _write(
        tmp_path,
        f"""
        video:
          input_path: "{video}"
        detector:
          model_path: "{weights}"
        tracker:
          config_path: "{tracker}"
        """,
    )

    with pytest.raises(ConfigError, match="tracker_type"):
        validate_runtime_paths(load_config(config_path))
