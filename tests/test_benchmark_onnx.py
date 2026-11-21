"""ONNX comparison metrics without ONNX Runtime or a video."""

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_onnx",
    ROOT / "scripts" / "benchmark_onnx.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules["benchmark_onnx"] = benchmark
SPEC.loader.exec_module(benchmark)


def test_matching_is_one_to_one_inside_a_class():
    reference = [_box(0, 0, 10, 10, cls=2, conf=0.9)]
    candidate = [
        _box(0, 0, 10, 10, cls=2, conf=0.91),
        _box(1, 1, 9, 9, cls=2, conf=0.4),
    ]
    matched = benchmark.match_boxes(reference, candidate)

    assert matched["matched"] == 1
    assert matched["unmatched_candidate"] == 1
    assert matched["pairs"][0]["iou"] == 1.0


def test_different_classes_and_low_overlap_stay_unmatched():
    reference = [_box(0, 0, 10, 10, cls=2, conf=0.8)]
    other_class = benchmark.match_boxes(
        reference,
        [_box(0, 0, 10, 10, cls=0, conf=0.8)],
    )
    distant = benchmark.match_boxes(
        reference,
        [_box(100, 100, 110, 110, cls=2, conf=0.8)],
    )

    assert other_class["matched"] == 0
    assert distant["matched"] == 0
    assert distant["unmatched_reference"] == 1
    assert distant["unmatched_candidate"] == 1


def test_summary_reports_near_threshold_differences():
    frames = [
        benchmark.match_boxes(
            [_box(0, 0, 10, 10, cls=2, conf=0.26)],
            [_box(0, 0, 10, 10, cls=2, conf=0.40)],
        ),
        benchmark.match_boxes(
            [],
            [_box(0, 0, 4, 4, cls=2, conf=0.27), _box(20, 20, 30, 30, cls=2, conf=0.9)],
        ),
    ]
    summary = benchmark.summarize_matches(frames)

    assert summary["matched"] == 1
    assert summary["unmatched_candidate"] == 2
    assert summary["minimum_iou"] == 1.0
    assert summary["median_center_px"] == 0.0
    assert summary["near_threshold_confidence_gaps"] == 1
    assert summary["near_threshold_unmatched"] == 1
    assert summary["maximum_confidence_abs_diff"] == pytest.approx(0.14)


def test_latency_summary_skips_warmup():
    rows = [
        _timing(True, 1000.0),
        _timing(False, 10.0),
        _timing(False, 30.0),
    ]
    summary = benchmark.summarize_latency(rows)

    assert summary["elapsed"]["mean_ms"] == 20.0
    assert summary["elapsed"]["count"] == 2
    assert summary["warmup_ms"] == 1000.0
    assert summary["steady_frames_per_second"] == 50.0


def test_percentiles_use_the_linear_method():
    stats = benchmark.distribution([10.0, 20.0, 30.0, 40.0])

    assert stats["p95_ms"] == 38.5
    assert stats["median_ms"] == 25.0


def test_export_settings_reject_embedded_nms_and_quantization():
    assert benchmark.validate_export_settings(benchmark.EXPORT_SETTINGS) == []
    assert "nms" in benchmark.validate_export_settings(
        {**benchmark.EXPORT_SETTINGS, "nms": True}
    )
    assert "dynamic" in benchmark.validate_export_settings(
        {**benchmark.EXPORT_SETTINGS, "dynamic": True}
    )
    assert "quantize" in benchmark.validate_export_settings(
        {**benchmark.EXPORT_SETTINGS, "quantize": 8}
    )
    assert "opset" in benchmark.validate_export_settings(
        {**benchmark.EXPORT_SETTINGS, "opset": True}
    )


def test_track_fragment_counts_a_return_after_a_gap():
    summary = benchmark.fragment_summary([[1], [1], [], [1], [2]])

    assert summary["unique_ids"] == 2
    assert summary["fragments"] == 3
    assert summary["ids_with_multiple_fragments"] == 1


def test_prediction_row_comparison_counts_category_changes():
    reference = [{"frame": "1", "track_id": "2", "class_name": "car", "lane_status": "IN_LANE", "approach_state": "STABLE", "evidence": "NO_APPROACH_EVIDENCE", "event_state": "NONE"}]
    candidate = [{**reference[0], "lane_status": "OUT_LANE"}]
    compared = benchmark.compare_prediction_rows(reference, candidate)

    assert compared["mismatched"] == 1
    assert compared["missing"] == 0


def test_missing_onnx_runtime_exits_before_a_video(monkeypatch, capsys):
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda name: None)

    code = benchmark.main([])
    error = capsys.readouterr().err

    assert code == 2
    assert "not installed" in error


def _box(x1, y1, x2, y2, *, cls: int, conf: float) -> dict:
    return {"cls": cls, "conf": conf, "xyxy": (x1, y1, x2, y2)}


def _timing(warmup: bool, elapsed: float) -> dict:
    return {
        "warmup": warmup,
        "elapsed_ms": elapsed,
        "ultralytics_preprocess_ms": 1.0,
        "ultralytics_inference_ms": 2.0,
        "ultralytics_postprocess_ms": 3.0,
    }
