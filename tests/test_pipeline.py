"""Pipeline orchestration with synthetic video and stand-in detectors."""

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import roadvision.pipeline as pipeline
from roadvision.annotate import (
    DISCLAIMER_BODY,
    LANE_COLORS,
    VehicleOverlay,
    annotate_frame,
    vehicle_lines,
)
from roadvision.config import (
    OBSERVATION_COLUMNS,
    ConfigError,
    RoadVisionConfig,
)
from roadvision.pipeline import (
    EVENT_COLUMNS,
    PipelineError,
    TrackedObject,
    run_video,
)
from src.lanes.temporal_filter import TemporalLaneFilter


ROOT = Path(__file__).resolve().parents[1]
FPS = 10.0


def _config(tmp_path, video, *, folder="out", max_frames=None, device="cpu"):
    # These tests replace the detector. The weights path only has to exist.
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"")
    base = RoadVisionConfig()
    return replace(
        base,
        video=replace(
            base.video,
            input_path=str(video),
            output_dir=str(tmp_path / folder),
            max_frames=max_frames,
        ),
        detector=replace(
            base.detector,
            device=device,
            model_path=str(weights),
        ),
        tracker=replace(
            base.tracker,
            config_path=str(ROOT / "configs" / "bytetrack_threshold.yaml"),
        ),
    )


def _frames(count, height=120, width=160):
    frames = []

    for index in range(count):
        frame = np.full((height, width, 3), 40 + index, np.uint8)
        frames.append(frame)

    return frames


def _write_clip(path, frames, fps=FPS):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create the synthetic test video")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _growing_object(frame: int) -> TrackedObject:
    height = math.exp((frame - 1) / FPS)
    return TrackedObject(
        bbox=(30.0, 40.0, 80.0, 40.0 + height),
        track_id=4,
        class_id=2,
        class_name="car",
        confidence=0.91,
    )


def _install_detector(monkeypatch, boxes_by_frame):
    created = []

    class ScriptedDetector:
        def __init__(self, config):
            created.append(self)
            self.n = 0

        def track(self, frame):
            self.n += 1
            return list(boxes_by_frame.get(self.n, []))

    monkeypatch.setattr(pipeline, "YoloByteTrack", ScriptedDetector)
    return created


def _install_in_lane(monkeypatch):
    seen = {}

    def conservative(bbox, left, right, margin=0.05):
        seen["margin"] = margin
        return "IN_LANE", "inside_lane", 0.2

    def detailed(bbox, polygon):
        return "IN_LANE", "inside_polygon"

    monkeypatch.setattr(
        pipeline,
        "classify_vehicle_lane_conservative",
        conservative,
    )
    monkeypatch.setattr(
        pipeline,
        "classify_vehicle_lane_detailed",
        detailed,
    )
    return seen


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _count_video_frames(path):
    capture = cv2.VideoCapture(str(path))
    count = 0

    try:
        while True:
            ok, _frame = capture.read()

            if not ok:
                break

            count += 1
    finally:
        capture.release()

    return count


class FakeCapture:
    def __init__(self, frames, fps=FPS, opened=True):
        self.frames = frames
        self.fps = fps
        self.opened = opened
        self.released = False
        self.index = 0
        self.width = 64
        self.height = 48

        if frames:
            self.height, self.width = frames[0].shape[:2]

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width

        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)

        return 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None

        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class MemoryWriter:
    latest = None

    def __init__(self, *args, **kwargs):
        self.released = False
        self.frames = []
        MemoryWriter.latest = self

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.shape)
        return True

    def release(self):
        self.released = True


def test_no_detections_still_updates_the_lane_filter(tmp_path, monkeypatch):
    frames = _frames(3)
    video = tmp_path / "empty.mp4"
    _write_clip(video, frames)
    _install_detector(monkeypatch, {})
    lane_frames = []
    original_update = TemporalLaneFilter.update

    def record_update(self, frame_number, lane_info):
        lane_frames.append(frame_number)
        return original_update(self, frame_number, lane_info)

    monkeypatch.setattr(pipeline.TemporalLaneFilter, "update", record_update)
    result = run_video(_config(tmp_path, video))

    assert lane_frames == [1, 2, 3]
    assert result.frames_processed == 3
    assert result.observation_count == 0
    assert result.episode_count == 0
    assert _count_video_frames(result.annotated_video) == 3
    assert _read_csv(result.observations_csv) == [list(OBSERVATION_COLUMNS)]
    assert _read_csv(result.events_csv) == [list(EVENT_COLUMNS)]


def test_multiple_tracked_objects_are_written(tmp_path, monkeypatch):
    video = tmp_path / "two.mp4"
    _write_clip(video, _frames(1))
    _install_detector(
        monkeypatch,
        {
            1: [
                TrackedObject((10, 30, 40, 70), 7, 2, "car", 0.8),
                TrackedObject((50, 30, 90, 80), 8, 7, "truck", 0.7),
            ]
        },
    )
    result = run_video(_config(tmp_path, video))
    rows = _read_csv(result.observations_csv)
    header = rows[0]
    body = rows[1:]

    assert header == list(OBSERVATION_COLUMNS)
    assert len(body) == 2
    assert {row[header.index("track_id")] for row in body} == {"7", "8"}
    assert {row[header.index("class_name")] for row in body} == {"car", "truck"}
    assert {row[header.index("lane_status")] for row in body} <= {
        "IN_LANE",
        "OUT_LANE",
        "UNKNOWN",
    }
    assert all(row[header.index("growth_robust_5")] == "" for row in body)
    assert all(row[header.index("event_state")] == "NONE" for row in body)


def test_max_frames_stops_early(tmp_path, monkeypatch):
    video = tmp_path / "long.mp4"
    _write_clip(video, _frames(4))
    created = _install_detector(monkeypatch, {})
    result = run_video(_config(tmp_path, video, max_frames=2))

    assert result.frames_processed == 2
    assert created[0].n == 2
    assert _count_video_frames(result.annotated_video) == 2


def test_repeated_runs_do_not_share_track_state(tmp_path, monkeypatch):
    video = tmp_path / "approach.mp4"
    _write_clip(video, _frames(8))
    boxes = {frame: [_growing_object(frame)] for frame in range(1, 9)}
    _install_detector(monkeypatch, boxes)
    _install_in_lane(monkeypatch)
    processors = []
    real_processor = pipeline.TrackMotionProcessor

    class RecordingProcessor(real_processor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            processors.append(self)

    monkeypatch.setattr(pipeline, "TrackMotionProcessor", RecordingProcessor)

    first = run_video(_config(tmp_path, video, folder="first"))
    second = run_video(
        _config(tmp_path, video, folder="second", max_frames=2)
    )

    assert first.episode_count == 1
    assert second.episode_count == 0
    assert len(processors) == 2
    assert processors[0] is not processors[1]
    assert processors[0]._tracks is not processors[1]._tracks
    assert processors[0]._finished
    assert processors[1]._finished

    events = _read_csv(first.events_csv)
    header = events[0]
    episode = events[1]

    assert header == list(EVENT_COLUMNS)
    assert episode[header.index("track_id")] == "4"
    assert episode[header.index("start_frame")] == "5"
    assert episode[header.index("confirmation_frame")] == "7"
    assert episode[header.index("end_frame")] == "8"
    assert _read_csv(second.events_csv) == [list(EVENT_COLUMNS)]


def test_confirmed_episode_uses_the_configured_margin(tmp_path, monkeypatch):
    video = tmp_path / "margin.mp4"
    _write_clip(video, _frames(8))
    _install_detector(
        monkeypatch,
        {frame: [_growing_object(frame)] for frame in range(1, 9)},
    )
    seen = _install_in_lane(monkeypatch)
    result = run_video(_config(tmp_path, video))
    rows = _read_csv(result.observations_csv)
    header = rows[0]
    states = [row[header.index("event_state")] for row in rows[1:]]

    assert seen["margin"] == 0.05
    assert states == [
        "NONE",
        "NONE",
        "NONE",
        "NONE",
        "PENDING",
        "PENDING",
        "CONFIRMED",
        "CONFIRMED",
    ]
    assert result.episode_count == 1


def test_metadata_records_the_run(tmp_path, monkeypatch):
    video = tmp_path / "meta.mp4"
    _write_clip(video, _frames(2))
    _install_detector(monkeypatch, {})
    result = run_video(_config(tmp_path, video))
    document = json.loads(result.metadata_json.read_text(encoding="utf-8"))

    assert document["schema_version"] == "1"
    assert document["model"]["device"] == "cpu"
    assert document["model"]["confidence"] == 0.25
    assert document["model"]["classes"] == [0, 1, 2, 3, 5, 7]
    assert document["tracker"]["settings"]["new_track_thresh"] == 0.40
    assert document["tracker"]["settings"]["track_buffer"] == 30
    assert document["video"]["frames_processed"] == 2
    assert document["video"]["width"] == 160
    assert document["video"]["height"] == 120
    assert document["video"]["fps"] == pytest.approx(FPS)
    assert document["outputs"]["observation_columns"] == list(OBSERVATION_COLUMNS)
    assert document["outputs"]["event_columns"] == list(EVENT_COLUMNS)
    assert document["formatting"]["motion_uses_unrounded_detector_boxes"] is True
    assert document["counts"]["observations"] == 0
    assert document["timing"]["elapsed_seconds"] >= 0.0
    assert "filtered_vp_y" not in document["outputs"]["observation_columns"]
    assert "motion_quality" not in document["outputs"]["observation_columns"]


def test_missing_video_is_rejected(tmp_path, monkeypatch):
    constructed = []

    class Unexpected:
        def __init__(self, config):
            constructed.append(config)

    monkeypatch.setattr(pipeline, "YoloByteTrack", Unexpected)

    with pytest.raises(ConfigError, match="Input video not found"):
        run_video(_config(tmp_path, tmp_path / "missing.mp4"))

    assert constructed == []


def test_unreadable_video_is_rejected(tmp_path):
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"not a video")

    with pytest.raises(PipelineError, match="Cannot open video"):
        run_video(_config(tmp_path, video))


def test_invalid_fps_releases_the_capture(tmp_path, monkeypatch):
    video = tmp_path / "fps.mp4"
    video.write_bytes(b"placeholder")
    capture = FakeCapture(_frames(1), fps=0.0)
    writers = []

    class GuardWriter(MemoryWriter):
        def __init__(self, *args, **kwargs):
            writers.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter", GuardWriter)

    with pytest.raises(PipelineError, match="Invalid input video FPS"):
        run_video(_config(tmp_path, video))

    assert capture.released
    assert writers == []


def test_failed_writer_releases_resources(tmp_path, monkeypatch):
    video = tmp_path / "writer.mp4"
    video.write_bytes(b"placeholder")
    capture = FakeCapture(_frames(2))
    constructed = []

    class DeadWriter:
        def __init__(self, *args, **kwargs):
            self.released = False
            DeadWriter.latest = self

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    class Unexpected:
        def __init__(self, config):
            constructed.append(config)

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter", DeadWriter)
    monkeypatch.setattr(pipeline, "YoloByteTrack", Unexpected)

    with pytest.raises(PipelineError, match="Cannot create video"):
        run_video(_config(tmp_path, video))

    assert capture.released
    assert DeadWriter.latest.released
    assert constructed == []


def test_detector_failure_closes_files_and_keeps_the_header(tmp_path, monkeypatch):
    video = tmp_path / "boom.mp4"
    video.write_bytes(b"placeholder")
    capture = FakeCapture(_frames(2))

    class Boom:
        def __init__(self, config):
            pass

        def track(self, frame):
            raise RuntimeError("detector failed")

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter", MemoryWriter)
    monkeypatch.setattr(pipeline, "YoloByteTrack", Boom)
    config = _config(tmp_path, video)

    with pytest.raises(RuntimeError, match="detector failed"):
        run_video(config)

    assert capture.released
    assert MemoryWriter.latest.released
    observations = Path(config.video.output_dir) / "observations.csv"
    assert _read_csv(observations) == [list(OBSERVATION_COLUMNS)]
    assert not (Path(config.video.output_dir) / "metadata.json").exists()


def test_each_run_constructs_a_new_model(tmp_path, monkeypatch):
    video = tmp_path / "model.mp4"
    _write_clip(video, _frames(1))

    class FakeYOLO:
        created = []

        def __init__(self, path):
            self.path = path
            self.calls = []
            FakeYOLO.created.append(self)

        def track(self, source, **kwargs):
            self.calls.append(kwargs)
            return [type("Result", (), {"boxes": None})()]

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    run_video(_config(tmp_path, video, folder="run-a", max_frames=1))
    run_video(_config(tmp_path, video, folder="run-b", max_frames=1))

    assert len(FakeYOLO.created) == 2
    assert FakeYOLO.created[0] is not FakeYOLO.created[1]

    for model in FakeYOLO.created:
        kwargs = model.calls[0]
        assert kwargs["persist"] is True
        assert kwargs["device"] == "cpu"
        assert kwargs["conf"] == 0.25
        assert kwargs["classes"] == [0, 1, 2, 3, 5, 7]
        assert kwargs["verbose"] is False
        assert str(kwargs["tracker"]).endswith("bytetrack_threshold.yaml")


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_unavailable_accelerator_does_not_fall_back(tmp_path, monkeypatch, device):
    import torch

    if device == "mps":
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    constructed = []

    class Unexpected:
        def __init__(self, config):
            constructed.append(config)

    monkeypatch.setattr(pipeline, "YoloByteTrack", Unexpected)

    with pytest.raises(PipelineError, match="does not fall back to CPU"):
        run_video(_config(tmp_path, tmp_path / "unused.mp4", device=device))

    assert constructed == []


def test_annotation_colors_and_disclaimer():
    frame = np.full((240, 320, 3), 255, np.uint8)
    vehicles = [
        VehicleOverlay(
            bbox=(40, 140, 110, 210),
            track_id=3,
            class_name="car",
            lane_status="IN_LANE",
            lane_reason="inside_lane",
            approach_state="APPROACHING",
            event_state="CONFIRMED",
            tau_proxy_s=1.25,
        )
    ]
    annotated = annotate_frame(frame, None, None, vehicles, 4, False)

    assert tuple(int(value) for value in annotated[170, 40]) == LANE_COLORS["IN_LANE"]
    assert tuple(int(value) for value in annotated[10, 8]) == (18, 18, 18)

    lines = [text for text, _color in vehicle_lines(vehicles[0])]

    assert "Scale: APPROACHING" in lines
    assert "Evidence: CONFIRMED" in lines
    assert "Inverse expansion: 1.25 s" in lines
    assert all("collision" not in text.lower() for text in lines)
    assert "not collision warnings" in DISCLAIMER_BODY

    held = VehicleOverlay(
        bbox=(40, 140, 110, 210),
        track_id=3,
        class_name="car",
        lane_status="UNKNOWN",
        lane_reason="lane_unavailable",
        approach_state="STABLE",
        event_state="HELD",
        tau_proxy_s=None,
    )
    held_lines = [text for text, _color in vehicle_lines(held)]

    assert "Previous approach evidence; lane uncertain" in held_lines
    assert LANE_COLORS["OUT_LANE"] == (120, 210, 120)
    assert LANE_COLORS["UNKNOWN"] == (180, 180, 180)
    assert LANE_COLORS["IN_LANE"] == (0, 220, 255)
