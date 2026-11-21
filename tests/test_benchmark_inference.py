"""Inference-diagnostic statistics without a video or model weights."""

import importlib.util
import sys
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_inference",
    ROOT / "scripts" / "benchmark_inference.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules["benchmark_inference"] = benchmark
SPEC.loader.exec_module(benchmark)


def test_steady_state_excludes_warmup_and_does_not_invent_tracking():
    rows = [
        _row(1, warmup=True, track=1000.0, callback=None, measured=False),
        _row(2, track=10.0, callback=1.0, measured=True),
        _row(3, track=30.0, callback=3.0, measured=True),
    ]
    summary = benchmark.summarize_rows(rows)

    assert summary["measured_frames"] == 2
    assert summary["warmup_track_call_ms"] == 1000.0
    assert summary["fields"]["track_call_ms"]["mean_ms"] == 20.0
    assert summary["fields"]["track_call_ms"]["median_ms"] == 20.0
    assert summary["tracker_callback_measured_frames"] == 2
    assert "bytetrack_ms" not in summary["fields"]


def test_empty_callback_samples_stay_empty():
    rows = [
        _row(1, warmup=True, track=5.0, callback=None, measured=False),
        _row(2, track=5.0, callback=None, measured=False),
    ]
    stats = benchmark.summarize_rows(rows)["fields"]["tracker_callback_ms"]

    assert stats["count"] == 0
    assert stats["mean_ms"] is None
    assert stats["p95_ms"] is None


def test_percentiles_follow_the_linear_method():
    stats = benchmark.distribution([10.0, 20.0, 30.0, 40.0])

    assert stats["p95_ms"] == 38.5
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 40.0


def test_tracker_probe_must_enclose_only_the_tracker_callback():
    def tracker(predictor, persist=False):
        return predictor

    start = lambda predictor: None
    end = lambda predictor: None
    noop = lambda predictor: None
    callback = partial(tracker)
    clean = [noop, start, callback, end]
    dirty = [start, callback, noop, end]

    assert benchmark.callback_bracket_is_exclusive(
        clean,
        start,
        end,
        tracker,
    )
    assert not benchmark.callback_bracket_is_exclusive(
        dirty,
        start,
        end,
        tracker,
    )


def test_probe_installation_rejects_a_split_callback_list():
    def tracker(predictor, persist=False):
        return predictor

    model = _model([lambda predictor: None, partial(tracker)])
    model.predictor = _model([])
    model.predictor.callbacks = {}
    probe, reason = benchmark.install_tracker_probe(model, tracker)

    assert probe is None
    assert "not the model callback list" in reason
    assert len(model.callbacks["on_predict_postprocess_end"]) == 2


def test_probe_installation_wraps_one_tracker_callback():
    def tracker(predictor, persist=False):
        return predictor

    model = _model([lambda predictor: None, partial(tracker)])
    model.predictor = model
    probe, reason = benchmark.install_tracker_probe(model, tracker)
    callbacks = model.callbacks["on_predict_postprocess_end"]

    assert probe is not None
    assert "only the ByteTrack" in reason
    assert benchmark.callback_bracket_is_exclusive(
        callbacks,
        probe.mark_start,
        probe.mark_end,
        tracker,
    )


def test_callback_source_check_requires_both_operations():
    def tracked(predictor):
        predictor.boxes.cpu().numpy()
        tracker.update(predictor)

    def other(predictor):
        return predictor

    confirmed = benchmark.describe_tracker_callback(tracked)
    missing = benchmark.describe_tracker_callback(other)

    assert confirmed == {
        "copies_boxes_to_cpu": True,
        "calls_tracker_update": True,
    }
    assert missing["copies_boxes_to_cpu"] is False
    assert missing["calls_tracker_update"] is False


def test_report_keeps_the_speed_gap_out_of_the_tracker_label():
    summary = _summary()
    report = benchmark.render_report(summary)

    assert "## Synchronized wall-clock timing" in report
    assert "## Ultralytics-reported internal timing" in report
    assert "## First-frame initialization" in report
    assert "## Tracking callback and tensor conversion" in report
    assert "## Timing ambiguity" in report
    assert "not a ByteTrack duration" in report
    assert "1000.000 ms" in report
    assert "mean 20.000 ms" in report


def test_timing_csv_preserves_a_missing_callback(tmp_path):
    path = tmp_path / "inference_timings.csv"
    benchmark.write_timings(
        path,
        [_row(1, warmup=True, track=4.0, callback=None, measured=False)],
    )
    text = path.read_text(encoding="utf-8")

    assert text.splitlines()[0] == ",".join(benchmark.TIMING_COLUMNS)
    assert ",," in text.splitlines()[1]


def _row(
    frame: int,
    *,
    warmup: bool = False,
    track: float,
    callback: float | None,
    measured: bool,
) -> dict[str, object]:
    return {
        "timing_mode": benchmark.OUTER_SYNC,
        "frame": frame,
        "warmup": warmup,
        "tracked_objects": 1,
        "track_call_ms": track,
        "ultralytics_preprocess_ms": 1.0,
        "ultralytics_inference_ms": 2.0,
        "ultralytics_postprocess_ms": 3.0,
        "tracker_callback_ms": callback,
        "tracker_callback_measured": measured,
        "result_cpu_conversion_ms": 0.5,
        "synced_preprocess_ms": None,
        "synced_inference_ms": None,
        "synced_postprocess_ms": None,
    }


def _summary() -> dict:
    rows = [
        _row(1, warmup=True, track=1000.0, callback=None, measured=False),
        _row(2, track=10.0, callback=1.0, measured=True),
        _row(3, track=30.0, callback=3.0, measured=True),
    ]
    outer = benchmark.summarize_rows(rows)
    outer.update(
        {
            "model_construction_ms": 12.0,
            "tracker_probe_reason": "Probes enclose only the ByteTrack callback.",
            "tracker_callback_body": {
                "copies_boxes_to_cpu": True,
                "calls_tracker_update": True,
            },
        }
    )
    stage_rows = [
        {
            **_row(2, track=12.0, callback=1.0, measured=True),
            "warmup": False,
            "synced_preprocess_ms": 4.0,
            "synced_inference_ms": 9.0,
            "synced_postprocess_ms": 2.0,
        }
    ]
    stage_rows.insert(
        0,
        {
            **_row(1, warmup=True, track=50.0, callback=None, measured=False),
            "synced_preprocess_ms": 1.0,
            "synced_inference_ms": 2.0,
            "synced_postprocess_ms": 1.0,
        },
    )
    stage = benchmark.summarize_rows(stage_rows)
    stage.update(
        {
            "model_construction_ms": 12.0,
            "tracker_probe_reason": outer["tracker_probe_reason"],
            "tracker_callback_body": outer["tracker_callback_body"],
        }
    )
    return {
        "environment": {
            "python": "3.12.16",
            "operating_system": "test",
            "cpu_architecture": "arm64",
            "torch": "2.15.0",
            "ultralytics": "8.4.193",
        },
        "settings": {
            "model_path": "yolo11n.pt",
            "device": "cpu",
            "confidence": 0.25,
            "iou": 0.7,
            "imgsz": 640,
            "classes": [0, 2],
            "tracker_path": "configs/bytetrack_threshold.yaml",
            "video_path": "clip.mp4",
            "frames_decoded": 2,
            "input_color": "BGR uint8 from OpenCV",
        },
        "passes": {
            benchmark.OUTER_SYNC: outer,
            benchmark.STAGE_SYNC: stage,
        },
        "synchronization_overhead": {
            "outer_track_call_mean_ms": 20.0,
            "stage_sync_track_call_mean_ms": 12.0,
            "added_mean_ms": -8.0,
        },
        "timing_definition": {
            "track_call_ms": "Synchronized wall clock around YOLO.track.",
            "ultralytics_speed": "Unsynchronized Ultralytics timings.",
            "warmup": "The first frame is excluded from steady-state statistics.",
            "tracker_callback_ms": "Tracker callback only.",
            "result_cpu_conversion_ms": "Wrapper conversion after track().",
            "stage_sync": "Extra synchronization around each stage.",
        },
    }


def _model(callbacks: list) -> object:
    class Model:
        pass

    model = Model()
    model.callbacks = {"on_predict_postprocess_end": list(callbacks)}
    model.predictor = None
    return model
