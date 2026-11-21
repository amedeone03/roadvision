"""Profiling analysis without a video or model weights."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_profiling",
    ROOT / "scripts" / "analyze_profiling.py",
)
analyze = importlib.util.module_from_spec(SPEC)
sys.modules["analyze_profiling"] = analyze
SPEC.loader.exec_module(analyze)


def test_warmup_is_excluded_and_stages_reconcile():
    rows = [
        _timing(1, warmup=True, inference=500.0, total=500.0),
        _timing(2, inference=6.0, lane=3.0, video=1.0, total=10.0),
        _timing(3, inference=8.0, lane=1.0, video=1.0, total=10.0),
    ]
    analysis = analyze.analyze_timings(rows, _summary(rows[1:]))

    assert analysis["measured_frames"] == 2
    assert analysis["warmup_frames"] == 1
    assert analysis["stages"]["frame_total_ms"]["total_ms"] == pytest.approx(20.0)
    assert analysis["stages"]["inference_tracking_ms"]["mean_ms"] == pytest.approx(7.0)
    assert analysis["reconciliation_max_abs_ms"] == pytest.approx(0.0)
    assert analysis["summary_max_abs_total_delta_ms"] == pytest.approx(0.0)
    shares = sum(item["percent"] for item in analysis["components"])
    assert shares == pytest.approx(100.0)
    inference = analysis["components"][0]
    assert inference["percent"] == pytest.approx(70.0)


def test_slowest_slice_uses_the_one_percent_ceiling():
    rows = [
        _timing(index, total=10.0 + (100.0 if index == 4 else 0.0))
        for index in range(1, 11)
    ]
    analysis = analyze.analyze_timings(rows, _summary(rows))

    assert analysis["slowest"]["count"] == 1
    assert analysis["slowest"]["frames"] == [4]
    assert analysis["slowest"]["chosen"]["mean_frame_total_ms"] == pytest.approx(
        110.0
    )


def test_lane_geometry_leaves_frames_without_observations_separate(tmp_path):
    rows = [
        _timing(1, lane=1.0, total=5.0),
        _timing(2, lane=3.0, total=5.0),
        _timing(3, lane=9.0, total=5.0),
    ]
    analysis = analyze.analyze_timings(rows, _summary(rows))
    path = _write_observations(
        tmp_path,
        [
            {"frame": "2", "lane_geometry_valid": "True"},
            {"frame": "3", "lane_geometry_valid": "False"},
        ],
    )
    analyze.attach_lane_geometry(analysis, rows, path)

    groups = analysis["lane_geometry"]["groups"]
    assert groups["unobserved"]["count"] == 1
    assert groups["true"]["mean_lane_detection_ms"] == pytest.approx(3.0)
    assert groups["false"]["mean_lane_detection_ms"] == pytest.approx(9.0)
    assert analysis["lane_geometry"]["conflicts"] == 0


def test_report_ranks_the_largest_component(tmp_path):
    rows = [
        _timing(1, warmup=True, inference=1.0, total=1.0),
        _timing(2, inference=8.0, lane=1.0, video=1.0, total=10.0),
        _timing(3, inference=6.0, lane=3.0, video=1.0, total=10.0),
    ]
    summary = _summary(rows[1:])
    summary["environment"] = {
        "python": "3.12.16",
        "implementation": "cpython",
        "operating_system": "test",
        "cpu_architecture": "arm64",
        "torch": "2.15.0",
        "ultralytics": "8.4.193",
    }
    summary["run"].update(
        {
            "device": "cpu",
            "model_path": "yolo11n.pt",
            "tracker_path": "configs/bytetrack_threshold.yaml",
            "confidence": 0.25,
            "classes": [0],
            "video_path": "clip.mp4",
            "width": 64,
            "height": 64,
            "fps": 10.0,
            "frames_processed": 2,
            "gpu_synchronization_enabled": False,
            "gpu_synchronization_calls": 0,
            "percentile_sample_sufficient": False,
            "pipeline_elapsed_seconds": 1.0,
            "pipeline_frames_per_second": 2.0,
        }
    )
    analysis = analyze.analyze_timings(rows, summary)
    report = analyze.render_report(
        analysis,
        summary,
        None,
        None,
        None,
        None,
        command="python -m roadvision run --profile",
        profile_dir=tmp_path,
        reference_dir=None,
    )

    assert "## Bottleneck ranking" in report
    assert report.index("1. Inference and tracking") < report.index(
        "2. Lane processing"
    )
    assert "not GPU kernel" in report
    assert "`inference_tracking_ms` | 14.0 |" in report


def test_output_comparison_requires_exact_categories(tmp_path):
    reference = tmp_path / "reference"
    profile = tmp_path / "profile"
    _write_pair(reference, lane="IN_LANE", event_end="10")
    _write_pair(profile, lane="OUT_LANE", event_end="10")

    comparison = analyze.compare_outputs(profile, reference)

    assert comparison["exact_agreement"] is False
    assert comparison["observations"]["mismatched_rows"] == 1
    assert comparison["events"]["mismatched_rows"] == 0

    _write_pair(profile, lane="IN_LANE", event_end="11")
    shifted = analyze.compare_outputs(profile, reference)
    assert shifted["events"]["missing"] == 1
    assert shifted["events"]["extra"] == 1


def test_main_writes_a_report_for_matching_outputs(tmp_path):
    profile = tmp_path / "profile"
    reference = tmp_path / "reference"
    profile.mkdir()
    _write_pair(reference, lane="IN_LANE", event_end="10")
    _write_pair(profile, lane="IN_LANE", event_end="10")
    rows = [_timing(1, inference=4.0, lane=4.0, video=2.0, total=10.0)]
    summary = _summary(rows)
    summary["environment"] = {
        "python": "3.12.16",
        "implementation": "cpython",
        "operating_system": "test",
        "cpu_architecture": "arm64",
        "torch": None,
        "ultralytics": None,
    }
    summary["run"].update(
        {
            "device": "cpu",
            "model_path": "yolo11n.pt",
            "tracker_path": "tracker.yaml",
            "confidence": 0.25,
            "classes": [2],
            "video_path": "clip.mp4",
            "width": 32,
            "height": 32,
            "fps": 10.0,
            "frames_processed": 1,
            "gpu_synchronization_enabled": False,
            "gpu_synchronization_calls": 0,
            "percentile_sample_sufficient": False,
            "pipeline_elapsed_seconds": 0.5,
            "pipeline_frames_per_second": 2.0,
        }
    )
    (profile / "profiling_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    _write_timings(profile / "frame_timings.csv", rows)
    report = tmp_path / "performance_report.md"

    code = analyze.main(
        [
            "--profile-dir",
            str(profile),
            "--reference-dir",
            str(reference),
            "--report",
            str(report),
        ]
    )

    text = report.read_text(encoding="utf-8")
    assert code == 0
    assert "Exact agreement" in text
    assert (tmp_path / "stage_share.svg").is_file()


def _timing(
    frame: int,
    *,
    warmup: bool = False,
    tracked: int = 1,
    inference: float = 0.0,
    lane: float = 0.0,
    video: float = 0.0,
    total: float = 0.0,
) -> dict[str, object]:
    other = total - inference - lane - video
    return {
        "frame": frame,
        "warmup": warmup,
        "tracked_objects": tracked,
        "lane_association_active": tracked > 0,
        "decode_ms": 0.0,
        "lane_detection_ms": lane,
        "lane_temporal_ms": 0.0,
        "inference_tracking_ms": inference,
        "lane_association_ms": 0.0,
        "motion_event_ms": 0.0,
        "annotation_ms": 0.0,
        "video_write_ms": video,
        "csv_write_ms": 0.0,
        "other_ms": other,
        "frame_total_ms": total,
    }


def _summary(measured: list[dict[str, object]]) -> dict:
    stages = {}
    for name in analyze.NUMERIC_STAGES:
        stages[name] = {
            "total_ms": sum(float(row[name]) for row in measured),
        }
    return {"stages": stages, "run": {}}


def _write_timings(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    fieldnames = [
        "frame",
        "warmup",
        "tracked_objects",
        "lane_association_active",
        *analyze.NUMERIC_STAGES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stored = dict(row)
            stored["warmup"] = "true" if row["warmup"] else "false"
            stored["lane_association_active"] = (
                "true" if row["lane_association_active"] else "false"
            )
            writer.writerow(stored)


def _write_observations(directory: Path, flags: list[dict[str, str]]) -> Path:
    import csv

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "observations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "track_id", "lane_geometry_valid"],
        )
        writer.writeheader()
        for index, row in enumerate(flags, start=1):
            writer.writerow(
                {
                    "frame": row["frame"],
                    "track_id": str(index),
                    "lane_geometry_valid": row["lane_geometry_valid"],
                }
            )
    return path


def _write_pair(directory: Path, *, lane: str, event_end: str) -> None:
    import csv

    directory.mkdir(parents=True, exist_ok=True)
    observation = {name: "" for name in analyze.EXACT_OBSERVATION_FIELDS}
    observation.update(
        {
            "frame": "1",
            "track_id": "2",
            "class_id": "2",
            "class_name": "car",
            "x1": "1.00",
            "y1": "2.00",
            "x2": "3.00",
            "y2": "4.00",
            "lane_status": lane,
            "approach_state": "STABLE",
            "evidence": "NO_APPROACH_EVIDENCE",
            "event_state": "NONE",
        }
    )
    with (directory / "observations.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "track_id", *analyze.EXACT_OBSERVATION_FIELDS],
        )
        writer.writeheader()
        writer.writerow(observation)
    with (directory / "events.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "track_id",
                "start_frame",
                "end_frame",
                "confirmation_frame",
                "duration_frames",
                "duration_s",
                "positive_frames",
                "interruption_frames",
                "min_tau_proxy_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "track_id": "2",
                "start_frame": "8",
                "end_frame": event_end,
                "confirmation_frame": "9",
                "duration_frames": "3",
                "duration_s": "0.1",
                "positive_frames": "3",
                "interruption_frames": "0",
                "min_tau_proxy_s": "1.0",
            }
        )
