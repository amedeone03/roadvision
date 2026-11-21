"""Full-video comparison helpers without a video or ONNX Runtime."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_coreml_full",
    ROOT / "scripts" / "validate_coreml_full.py",
)
validation = importlib.util.module_from_spec(SPEC)
sys.modules["validate_coreml_full"] = validation
SPEC.loader.exec_module(validation)


def test_frame_counts_include_frames_with_no_observations():
    reference = [_observation(1, 1)]
    candidate = [_observation(2, 9)]
    delta = validation.frame_count_delta(reference, candidate, frame_count=2)

    assert delta["frames_with_different_counts"] == 2
    assert delta["max_abs_count_delta"] == 1
    assert delta["reference_observations"] == 1
    assert delta["candidate_observations"] == 1


def test_matched_categories_ignore_track_ids():
    reference = [_observation(1, 1, lane="IN_LANE", approach="STABLE")]
    candidate = [_observation(1, 9, lane="OUT_LANE", approach="STABLE")]
    categories = validation.category_agreement(reference, candidate)

    assert categories["lane_status"]["mismatched"] == 1
    assert categories["approach_state"]["mismatched"] == 0
    assert categories["evidence"]["mismatched"] == 0


def test_same_boxes_with_new_ids_remain_one_trajectory():
    reference = [_observation(1, 1), _observation(2, 1)]
    candidate = [_observation(1, 4), _observation(2, 4)]
    tracking = validation.tracking_agreement(reference, candidate)

    assert tracking["mapped"] == 1
    assert tracking["renumbered"] == 1
    assert tracking["unmatched_reference"] == 0
    assert tracking["duration"]["max_abs_frame_delta"] == 0


def test_episode_windows_report_a_tau_change_without_a_new_episode():
    reference = [_episode("10", 81, 83, tau="0.648")]
    candidate = [_episode("8", 81, 83, tau="0.641")]
    compared = validation.episode_agreement(reference, candidate)

    assert compared["mapped"] == 1
    assert compared["same_bounds"] == 1
    assert compared["unmatched_reference"] == []
    assert compared["pairs"][0]["fields"]["min_tau_proxy_s"]["equal"] is False
    assert compared["pairs"][0]["fields"]["interruption_frames"]["equal"] is True


def _observation(frame: int, track_id: int, *, lane: str = "IN_LANE", approach: str = "STABLE") -> dict:
    return {
        "frame": str(frame),
        "track_id": str(track_id),
        "class_id": "2",
        "class_name": "car",
        "confidence": "0.9000",
        "x1": "0",
        "y1": "0",
        "x2": "10",
        "y2": "10",
        "lane_status": lane,
        "approach_state": approach,
        "evidence": "NO_APPROACH_EVIDENCE",
        "event_state": "NONE",
        "tau_proxy_s": "",
    }


def _episode(track_id: str, start: int, end: int, *, tau: str) -> dict:
    return {
        "track_id": track_id,
        "start_frame": str(start),
        "end_frame": str(end),
        "confirmation_frame": str(end),
        "duration_frames": str(end - start + 1),
        "positive_frames": str(end - start + 1),
        "interruption_frames": "0",
        "min_tau_proxy_s": tau,
    }
