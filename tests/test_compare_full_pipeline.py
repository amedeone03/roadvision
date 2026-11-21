"""Comparison rules for the full-video regression."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_full_pipeline",
    ROOT / "scripts" / "compare_full_pipeline.py",
)
compare = importlib.util.module_from_spec(SPEC)
sys.modules["compare_full_pipeline"] = compare
SPEC.loader.exec_module(compare)


def test_difference_summary_uses_mean_median_p95_and_max():
    summary = compare.summarize([1, 2, 3, 4, 100])

    assert summary["count"] == 5
    assert summary["mean"] == 22
    assert summary["median"] == 3
    assert summary["max"] == 100
    assert summary["p95"] == pytest.approx(80.8)
    assert compare.summarize([])["count"] == 0


def test_tracking_reports_key_overlap_without_equating_identity():
    reference = [
        _row(1, 2, box=(0, 0, 10, 10)),
        _row(2, 2, box=(0, 0, 10, 12)),
        _row(3, 7, box=(5, 5, 15, 15)),
    ]
    live = [
        _row(1, 2, box=(0, 0, 10, 11)),
        _row(2, 2, box=(0, 0, 10, 12)),
        _row(4, 9, box=(1, 1, 4, 4)),
    ]

    result = compare.compare_runs(reference, live, [], [])
    tracking = result["tracking"]

    assert tracking["shared_keys"] == 2
    assert tracking["missing_keys"] == 1
    assert tracking["added_keys"] == 1
    assert tracking["missing_examples"] == [(3, 7)]
    assert tracking["added_examples"] == [(4, 9)]
    assert tracking["reference_tracks"] == [2, 7]
    assert tracking["new_tracks"] == [2, 9]
    assert tracking["reference_frames"] == 3
    assert tracking["new_frames"] == 3
    assert tracking["max_corner"]["max"] == pytest.approx(1)
    assert tracking["iou"]["count"] == 2
    assert tracking["iou"]["mean"] < 1


def test_lane_matrix_separates_definite_and_uncertain_changes():
    reference = [
        _row(1, 1, lane="IN_LANE"),
        _row(2, 1, lane="OUT_LANE"),
        _row(3, 1, lane="UNKNOWN"),
        _row(4, 1, lane="IN_LANE"),
    ]
    live = [
        _row(1, 1, lane="IN_LANE"),
        _row(2, 1, lane="IN_LANE"),
        _row(3, 1, lane="OUT_LANE"),
        _row(4, 1, lane="UNKNOWN"),
    ]

    lanes = compare.compare_runs(reference, live, [], [])["lanes"]

    assert lanes["matches"] == 1
    assert lanes["agreement"] == 0.25
    assert lanes["matrix"]["OUT_LANE"]["IN_LANE"] == 1
    assert lanes["matrix"]["UNKNOWN"]["OUT_LANE"] == 1
    assert lanes["matrix"]["IN_LANE"]["UNKNOWN"] == 1
    assert lanes["definiteness"]["unchanged_definite"] == 1
    assert lanes["definiteness"]["definite_label_changed"] == 1
    assert lanes["definiteness"]["uncertain_to_definite"] == 1
    assert lanes["definiteness"]["definite_to_uncertain"] == 1


def test_growth_comparison_keeps_missing_values_separate():
    reference = [
        _row(1, 1, growth=0.20, approach="APPROACHING"),
        _row(2, 1, growth=0.10, approach="STABLE"),
        _row(3, 1, growth="", approach="UNKNOWN"),
        _row(4, 1, growth=0.40, approach="APPROACHING"),
    ]
    live = [
        _row(1, 1, growth=0.25, approach="APPROACHING"),
        _row(2, 1, growth=0.16, approach="APPROACHING"),
        _row(3, 1, growth="", approach="UNKNOWN"),
        _row(4, 1, growth="", approach="UNKNOWN"),
    ]

    motion = compare.compare_runs(reference, live, [], [])["motion"]

    assert motion["both_finite"] == 2
    assert motion["both_missing"] == 1
    assert motion["reference_missing_only"] == 0
    assert motion["new_missing_only"] == 1
    assert motion["growth"]["mean"] == pytest.approx(0.055)
    assert motion["growth"]["max"] == pytest.approx(0.06)
    assert motion["threshold_crossings"] == 1


def test_negative_threshold_crossing_is_counted_when_boxes_match():
    reference = [_row(1, 1, growth=-0.150109, approach="RECEDING")]
    live = [_row(1, 1, growth=-0.14942, approach="STABLE")]

    motion = compare.compare_runs(reference, live, [], [])["motion"]

    assert motion["threshold_crossings"] == 1
    assert motion["approach"]["matrix"]["RECEDING"]["STABLE"] == 1
    assert motion["growth"]["max"] == pytest.approx(0.000689)


def test_event_states_on_shared_keys_ignore_extra_tracks():
    reference = [
        _row(1, 1, event="NONE"),
        _row(2, 1, event="CONFIRMED"),
    ]
    live = [
        _row(1, 1, event="NONE"),
        _row(9, 4, event="HELD"),
    ]

    result = compare.compare_runs(reference, live, [], [])

    assert result["event_states"]["compared"] == 1
    assert result["event_states"]["matches"] == 1
    assert result["distributions"]["reference"]["event_state"]["CONFIRMED"] == 1
    assert result["distributions"]["new"]["event_state"]["HELD"] == 1


def test_episode_relations_keep_identity_shifts_separate():
    exact = _episode(2, 38, 42, 46, min_tau=2.0)
    exact_live = _episode(2, 38, 42, 46, min_tau=2.2)
    shifted_reference = _episode(3, 10, 12, 12)
    shifted_live = _episode(3, 10, 12, 14)
    split_reference = _episode(4, 10, 12, 20)
    split_live_a = _episode(4, 10, 12, 14)
    split_live_b = _episode(4, 16, 18, 20)
    merged_reference_a = _episode(5, 30, 32, 34)
    merged_reference_b = _episode(5, 36, 38, 40)
    merged_live = _episode(5, 30, 32, 40)
    ambiguous_reference_a = _episode(6, 1, 2, 10)
    ambiguous_reference_b = _episode(6, 8, 9, 20)
    ambiguous_live_a = _episode(6, 1, 2, 9)
    ambiguous_live_b = _episode(6, 9, 10, 20)
    identity_reference = _episode(7, 50, 52, 55)
    identity_live = _episode(8, 50, 52, 55)
    missing = _episode(9, 1, 2, 3)
    added = _episode(9, 20, 21, 22)
    rows = [
        _row(frame, 3)
        for frame in range(10, 15)
    ]

    result = compare.compare_runs(
        rows,
        rows,
        [
            exact,
            shifted_reference,
            split_reference,
            merged_reference_a,
            merged_reference_b,
            ambiguous_reference_a,
            ambiguous_reference_b,
            identity_reference,
            missing,
        ],
        [
            exact_live,
            shifted_live,
            split_live_a,
            split_live_b,
            merged_live,
            ambiguous_live_a,
            ambiguous_live_b,
            identity_live,
            added,
        ],
    )
    episodes = result["episodes"]

    assert len(episodes["exact"]) == 1
    assert episodes["boundary_matches"] == {
        "pairs": 2,
        "start": 2,
        "confirmation": 2,
        "end": 1,
        "all_three": 1,
    }
    assert episodes["exact_min_tau"]["max"] == pytest.approx(0.2)
    assert len(episodes["shifted"]) == 1
    assert "same rounded boxes" in episodes["shifted"][0]["attribution"]
    assert len(episodes["split"]) == 1
    assert len(episodes["merged"]) == 1
    assert len(episodes["ambiguous"]) == 1
    assert len(episodes["identity_candidates"]) == 1
    assert episodes["identity_candidates"][0]["reference"].track_id == 7
    assert episodes["identity_candidates"][0]["new"].track_id == 8
    assert [episode.track_id for episode in episodes["missing"]] == [9]
    assert [episode.track_id for episode in episodes["added"]] == [9]
    assert episodes["exact"][0]["reference"][0].track_id == 2


def test_shifted_episode_reports_a_lane_change_separately_from_growth():
    reference_rows = [_row(10, 3, lane="IN_LANE")]
    live_rows = [_row(10, 3, lane="UNKNOWN")]

    episodes = compare.compare_runs(
        reference_rows,
        live_rows,
        [_episode(3, 10, 10, 10)],
        [_episode(3, 10, 10, 11)],
    )["episodes"]

    assert "lane labels differ" in episodes["shifted"][0]["attribution"]


def test_adjacent_same_track_episodes_are_not_treated_as_one_event():
    episodes = compare.compare_runs(
        [],
        [],
        [_episode(1, 1, 2, 5)],
        [_episode(1, 6, 7, 10)],
    )["episodes"]

    assert episodes["shifted"] == []
    assert len(episodes["missing"]) == 1
    assert len(episodes["added"]) == 1
    assert episodes["identity_candidates"] == []


def test_reference_approach_alias_is_read_from_csv(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text(
        "frame,track_id,approach_t015,growth_robust_5\n"
        "1,2,STABLE,\n",
        encoding="utf-8",
    )

    rows = compare.read_observation_rows(path, approach_column="approach_t015")

    assert rows[0]["approach_state"] == "STABLE"
    assert compare._finite(rows[0]["growth_robust_5"]) is None


def test_identical_boxes_have_unit_iou_and_disjoint_boxes_have_zero():
    assert compare.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert compare.box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0


def test_main_writes_a_report_for_a_valid_tiny_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    _write_observation(output / "observations.csv", approach="approach_state")
    _write_observation(reference / "observations.csv", approach="approach_t015")
    header = (
        "track_id,start_frame,end_frame,confirmation_frame,"
        "duration_frames,duration_s,positive_frames,"
        "interruption_frames,min_tau_proxy_s\n"
    )
    (output / "events.csv").write_text(header, encoding="utf-8")
    (reference / "events.csv").write_text(header, encoding="utf-8")
    _write_clip(output / "annotated.mp4", value=20)
    _write_clip(tmp_path / "source.mp4", value=200)
    metadata = {
        "video": {
            "path": str(tmp_path / "source.mp4"),
            "width": 16,
            "height": 16,
            "fps": 10.0,
            "container_frame_count": 2,
            "frames_processed": 2,
        },
        "counts": {
            "observations": 1,
            "episodes": 0,
            "accepted_lane_frames": 1,
        },
        "timing": {"elapsed_seconds": 0.5, "frames_per_second": 4.0},
        "model": {
            "path": "yolo11n.pt",
            "device": "cpu",
            "confidence": 0.25,
            "classes": [0, 2],
        },
        "formatting": {"motion_uses_unrounded_detector_boxes": True},
        "outputs": {
            "annotated_video": "annotated.mp4",
            "observations_csv": "observations.csv",
            "events_csv": "events.csv",
            "metadata_json": "metadata.json",
        },
    }
    (output / "metadata.json").write_text(
        __import__("json").dumps(metadata),
        encoding="utf-8",
    )

    status = compare.main([
        "--output-dir",
        str(output),
        "--reference-observations",
        str(reference / "observations.csv"),
        "--reference-events",
        str(reference / "events.csv"),
        "--wall-clock-seconds",
        "0.7",
    ])

    report = (output / "comparison_report.md").read_text(encoding="utf-8")
    assert status == 0
    assert "not real time" in report
    assert "not collision warnings" in report
    assert "Shared keys: 1" in report


def _row(
    frame,
    track,
    lane="IN_LANE",
    growth=0.2,
    approach="APPROACHING",
    event="NONE",
    box=(0, 0, 10, 10),
    evidence="EGO_LANE_APPROACH",
):
    x1, y1, x2, y2 = box
    return {
        "frame": frame,
        "track_id": track,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "lane_status": lane,
        "growth_robust_5": growth,
        "approach_state": approach,
        "event_state": event,
        "evidence": evidence,
        "time_seconds": (frame - 1) / 10,
    }


def _episode(track, start, confirmation, end, min_tau=None):
    return compare.Episode(
        track_id=track,
        start_frame=start,
        end_frame=end,
        confirmation_frame=confirmation,
        duration_frames=end - start + 1,
        positive_frames=1,
        interruption_frames=0,
        duration_s=None,
        min_tau_proxy_s=min_tau,
    )


def _write_observation(path: Path, approach: str):
    from roadvision.config import OBSERVATION_COLUMNS

    values = {column: "" for column in OBSERVATION_COLUMNS}
    values.update({
        "frame": "1",
        "timestamp": "0.000",
        "track_id": "4",
        "class_id": "2",
        "class_name": "car",
        "confidence": "0.9000",
        "x1": "0.00",
        "y1": "0.00",
        "x2": "10.00",
        "y2": "10.00",
        "contact_x": "5.00",
        "contact_y": "10.00",
        "lane_status": "IN_LANE",
        "lane_reason": "inside",
        "baseline_status": "IN_LANE",
        "baseline_reason": "inside",
        "normalized_lateral_margin": "0.20000",
        "lane_valid": "True",
        "lane_geometry_valid": "True",
        "lane_temporal_reason": "consistent",
        "height": "10.0",
        "time_seconds": "0.0",
        "growth_robust_5": "",
        "approach_state": "UNKNOWN",
        "tau_proxy_s": "",
        "evidence": "MOTION_UNAVAILABLE",
        "event_state": "NONE",
    })
    if approach == "approach_t015":
        values["approach_t015"] = values.pop("approach_state")
        columns = [
            "approach_t015" if column == "approach_state" else column
            for column in OBSERVATION_COLUMNS
        ]
    else:
        columns = list(OBSERVATION_COLUMNS)
    path.write_text(
        ",".join(columns) + "\n" + ",".join(values[column] for column in columns) + "\n",
        encoding="utf-8",
    )


def _write_clip(path: Path, value: int):
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (16, 16),
    )
    assert writer.isOpened()
    frame = np.full((16, 16, 3), value, np.uint8)
    writer.write(frame)
    writer.write(frame)
    writer.release()
