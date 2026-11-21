"""Optional frame profiling without a real video or model weights."""

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import roadvision.pipeline as pipeline
from roadvision.config import OBSERVATION_COLUMNS, RoadVisionConfig
from roadvision.pipeline import TrackedObject, run_video
from roadvision.profiling import (
    FRAME_COLUMNS,
    FRAME_TIMINGS_NAME,
    PERCENTILE_MINIMUM_FRAMES,
    PROFILING_SUMMARY_NAME,
    STAGE_NAMES,
    FrameProfiler,
    build_summary,
    synchronize_device,
)


ROOT = Path(__file__).resolve().parents[1]


def test_span_measures_elapsed_wall_time():
    profiler = FrameProfiler(enabled=True, device="cpu")
    profiler.begin_frame()

    with profiler.span("decode_ms"):
        time.sleep(0.02)

    profiler.end_frame(frame=1, tracked_objects=0, lane_association_active=False)
    measured = profiler.frames[0]["decode_ms"]

    assert measured >= 15.0
    assert measured < 200.0
    assert profiler.frames[0]["warmup"] is True


def test_stage_times_aggregate_and_percentiles_skip_warmup():
    profiler = FrameProfiler(enabled=True, device="cpu")
    samples = [10.0, 20.0, 30.0, 40.0]

    for index, sample in enumerate(samples, start=1):
        profiler.frames.append(_frame(index, decode_ms=sample))

    summary = build_summary(
        profiler,
        {"frames_processed": 4, "pipeline_elapsed_seconds": 2.0},
    )
    decode = summary["stages"]["decode_ms"]
    measured = samples[1:]

    assert summary["run"]["warmup_frames"] == 1
    assert summary["run"]["measured_frames"] == 3
    assert summary["run"]["percentile_sample_sufficient"] is False
    assert decode["count"] == 3
    assert decode["total_ms"] == pytest.approx(sum(measured))
    assert decode["mean_ms"] == pytest.approx(30.0)
    assert decode["median_ms"] == pytest.approx(30.0)
    assert decode["min_ms"] == pytest.approx(20.0)
    assert decode["max_ms"] == pytest.approx(40.0)
    assert decode["p95_ms"] == pytest.approx(
        float(np.percentile(np.asarray(measured), 95, method="linear"))
    )
    assert set(summary["stages"]) == {*STAGE_NAMES, "other_ms", "frame_total_ms"}
    assert summary["run"]["pipeline_frames_per_second"] == pytest.approx(2.0)
    assert summary["run"]["percentile_minimum_frames"] == PERCENTILE_MINIMUM_FRAMES


def test_empty_profile_has_null_percentiles():
    summary = build_summary(
        FrameProfiler(enabled=True, device="cpu"),
        {"frames_processed": 0, "pipeline_elapsed_seconds": 0.0},
    )

    assert summary["stages"]["decode_ms"]["count"] == 0
    assert summary["stages"]["decode_ms"]["p95_ms"] is None
    assert summary["stages"]["frame_total_ms"]["mean_ms"] is None
    assert summary["run"]["measured_frames"] == 0
    assert summary["run"]["gpu_synchronization_enabled"] is False


def test_disabled_profiler_does_not_time_or_synchronize(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "roadvision.profiling.synchronize_device",
        lambda device: calls.append(device) or True,
    )
    profiler = FrameProfiler(enabled=False, device="mps")
    profiler.begin_frame()

    with profiler.span("decode_ms"):
        pass

    profiler.synchronize()
    profiler.end_frame(frame=1, tracked_objects=0, lane_association_active=False)

    assert profiler.frames == []
    assert calls == []
    assert profiler.synchronization_calls == 0


def test_enabled_accelerator_synchronizes_once_per_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "roadvision.profiling.synchronize_device",
        lambda device: calls.append(device) or True,
    )
    first = FrameProfiler(enabled=True, device="mps")
    second = FrameProfiler(enabled=True, device="cuda")
    first.synchronize()
    second.synchronize()

    assert calls == ["mps", "cuda"]
    assert first.synchronization_calls == 1
    assert second.frames == []
    assert first.frames == []


def test_cpu_synchronization_does_not_touch_torch():
    assert synchronize_device("cpu") is False


def test_mps_and_cuda_synchronization_call_torch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "torch.mps.synchronize",
        lambda: calls.append("mps"),
        raising=False,
    )
    monkeypatch.setattr(
        "torch.cuda.synchronize",
        lambda: calls.append("cuda"),
    )

    assert synchronize_device("mps") is True
    assert synchronize_device("cuda") is True
    assert calls == ["mps", "cuda"]


def test_timing_files_use_the_published_schema(tmp_path):
    profiler = FrameProfiler(enabled=True, device="cpu")
    profiler.frames.append(_frame(1, decode_ms=5.0, warmup=True))
    profiler.frames.append(
        _frame(2, decode_ms=8.0, tracked_objects=2, active=True)
    )
    timings, summary_path = profiler.write(
        tmp_path,
        {
            "frames_processed": 2,
            "pipeline_elapsed_seconds": 1.5,
            "model_path": "yolo11n.pt",
            "width": 1920,
            "height": 1080,
            "fps": 29.97,
            "video_path": "clip.mp4",
        },
    )
    with timings.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert timings.name == FRAME_TIMINGS_NAME
    assert summary_path.name == PROFILING_SUMMARY_NAME
    assert list(rows[0]) == list(FRAME_COLUMNS)
    assert rows[0]["warmup"] == "true"
    assert rows[1]["lane_association_active"] == "true"
    assert rows[1]["tracked_objects"] == "2"
    assert summary["schema_version"] == "1"
    assert summary["stages"]["decode_ms"]["count"] == 1
    assert summary["stages"]["decode_ms"]["mean_ms"] == pytest.approx(8.0)
    assert summary["timing_definition"]["gpu_stage"] == "inference_tracking_ms"
    assert "kernel" in summary["timing_definition"]["gpu_measurement"]
    assert "metadata.json" in summary["timing_definition"]["after_pipeline_clock"]
    assert "events.csv" in summary["timing_definition"]["outside_frame_stages"]
    assert summary["environment"]["python"]
    assert summary["run"]["width"] == 1920


def test_two_profilers_do_not_share_frames():
    first = FrameProfiler(enabled=True, device="cpu")
    second = FrameProfiler(enabled=True, device="cpu")
    first.frames.append(_frame(1, decode_ms=1.0))

    assert second.frames == []
    assert len(first.frames) == 1


def test_disabled_run_does_not_write_profile_files(tmp_path, monkeypatch):
    video = tmp_path / "plain.mp4"
    _write_clip(video, _frames(2))
    _install_detector(monkeypatch, {})
    result = run_video(_config(tmp_path, video))

    assert result.frame_timings_csv is None
    assert result.profiling_summary_json is None
    assert not (tmp_path / "out" / FRAME_TIMINGS_NAME).exists()
    assert not (tmp_path / "out" / PROFILING_SUMMARY_NAME).exists()
    assert _header(result.observations_csv) == list(OBSERVATION_COLUMNS)


def test_profiled_run_matches_an_unprofiled_run(tmp_path, monkeypatch):
    frames = _frames(3)
    boxes = {
        2: [TrackedObject((10, 20, 40, 70), 4, 2, "car", 0.9)],
    }
    plain_video = tmp_path / "plain.mp4"
    timed_video = tmp_path / "timed.mp4"
    _write_clip(plain_video, frames)
    _write_clip(timed_video, frames)
    _install_detector(monkeypatch, boxes)
    plain = run_video(_config(tmp_path, plain_video, folder="plain"))
    timed = run_video(
        _config(tmp_path, timed_video, folder="timed"),
        profile=True,
    )

    assert plain.observations_csv.read_text(encoding="utf-8") == (
        timed.observations_csv.read_text(encoding="utf-8")
    )
    assert plain.events_csv.read_text(encoding="utf-8") == (
        timed.events_csv.read_text(encoding="utf-8")
    )
    assert "profiling" not in plain.metadata_json.read_text(encoding="utf-8")
    assert "profiling" not in timed.metadata_json.read_text(encoding="utf-8")
    with timed.frame_timings_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(
        timed.profiling_summary_json.read_text(encoding="utf-8")
    )

    assert [row["frame"] for row in rows] == ["1", "2", "3"]
    assert rows[1]["lane_association_active"] == "true"
    assert rows[0]["lane_association_active"] == "false"
    assert rows[0]["motion_event_ms"] == "0.000000"
    assert summary["run"]["gpu_synchronization_enabled"] is False
    assert summary["run"]["gpu_synchronization_calls"] == 0
    assert summary["run"]["measured_frames"] == 2
    assert summary["stages"]["frame_total_ms"]["count"] == 2
    _assert_summary_matches_rows(summary, rows)


def test_profile_failure_closes_outputs_without_a_summary(tmp_path, monkeypatch):
    video = tmp_path / "boom.mp4"
    _write_clip(video, _frames(1))

    class Boom:
        def __init__(self, config):
            pass

        def track(self, frame):
            raise RuntimeError("detector failed")

    monkeypatch.setattr(pipeline, "YoloByteTrack", Boom)

    with pytest.raises(RuntimeError, match="detector failed"):
        run_video(_config(tmp_path, video, folder="boom"), profile=True)

    observations = tmp_path / "boom" / "observations.csv"
    assert observations.read_text(encoding="utf-8").splitlines() == [
        ",".join(OBSERVATION_COLUMNS)
    ]
    assert not (tmp_path / "boom" / PROFILING_SUMMARY_NAME).exists()
    assert not (tmp_path / "boom" / FRAME_TIMINGS_NAME).exists()


def test_profile_flag_is_forwarded(monkeypatch, tmp_path):
    from roadvision.cli import main
    from roadvision.pipeline import PipelineResult

    seen = []

    def fake(config, profile=False):
        seen.append(profile)
        output = Path(config.video.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        metadata = output / "metadata.json"
        metadata.write_text(
            json.dumps({"video": {"fps": 10, "width": 1, "height": 1}}),
            encoding="utf-8",
        )
        return PipelineResult(
            annotated_video=output / "annotated.mp4",
            observations_csv=output / "observations.csv",
            events_csv=output / "events.csv",
            metadata_json=metadata,
            frames_processed=1,
            observation_count=0,
            episode_count=0,
            elapsed_seconds=0.1,
            frame_timings_csv=output / FRAME_TIMINGS_NAME,
            profiling_summary_json=output / PROFILING_SUMMARY_NAME,
        )

    monkeypatch.setattr("roadvision.cli.run_video", fake)

    assert main(["run", "--profile", "--output", str(tmp_path)]) == 0
    assert seen == [True]


def _frame(
    number: int,
    *,
    decode_ms: float,
    warmup: bool | None = None,
    tracked_objects: int = 0,
    active: bool = False,
) -> dict:
    record = {
        "frame": number,
        "warmup": number == 1 if warmup is None else warmup,
        "tracked_objects": tracked_objects,
        "lane_association_active": active,
        "other_ms": 0.0,
        "frame_total_ms": decode_ms,
    }

    for name in STAGE_NAMES:
        record[name] = decode_ms if name == "decode_ms" else 0.0

    return record


def _assert_summary_matches_rows(summary: dict, rows: list[dict[str, str]]) -> None:
    measured = [row for row in rows if row["warmup"] == "false"]

    for name in (*STAGE_NAMES, "other_ms", "frame_total_ms"):
        values = [float(row[name]) for row in measured]
        stage = summary["stages"][name]
        assert stage["count"] == len(values)
        assert stage["total_ms"] == pytest.approx(sum(values), abs=1e-4)
        assert stage["min_ms"] == pytest.approx(min(values), abs=1e-4)
        assert stage["max_ms"] == pytest.approx(max(values), abs=1e-4)


def _config(tmp_path, video, *, folder="out"):
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"")
    base = RoadVisionConfig()
    return replace(
        base,
        video=replace(
            base.video,
            input_path=str(video),
            output_dir=str(tmp_path / folder),
            max_frames=None,
        ),
        detector=replace(
            base.detector,
            device="cpu",
            model_path=str(weights),
        ),
        tracker=replace(
            base.tracker,
            config_path=str(ROOT / "configs" / "bytetrack_threshold.yaml"),
        ),
    )


def _frames(count: int):
    return [
        np.full((48, 64, 3), 30 + index, np.uint8)
        for index in range(count)
    ]


def _write_clip(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    assert writer.isOpened()

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _install_detector(monkeypatch, boxes_by_frame) -> None:
    class ScriptedDetector:
        def __init__(self, config):
            self.n = 0

        def track(self, frame):
            self.n += 1
            return list(boxes_by_frame.get(self.n, []))

    monkeypatch.setattr(pipeline, "YoloByteTrack", ScriptedDetector)


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))
