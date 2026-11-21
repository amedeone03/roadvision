"""Core ML comparison helpers without ONNX Runtime or a video."""

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_coreml",
    ROOT / "scripts" / "benchmark_coreml.py",
)
coreml = importlib.util.module_from_spec(SPEC)
sys.modules["benchmark_coreml"] = coreml
SPEC.loader.exec_module(coreml)


def test_provider_options_match_this_runtime():
    assert coreml.validate_provider_options(coreml.COREML_OPTIONS) == []
    assert "MLComputeUnits" in coreml.validate_provider_options(
        {**coreml.COREML_OPTIONS, "MLComputeUnits": "CPUAndTPU"}
    )
    assert "ModelFormat" in coreml.validate_provider_options(
        {**coreml.COREML_OPTIONS, "ModelFormat": "FP16"}
    )


def test_capability_line_counts_unsupported_nodes():
    log = (
        "CoreMLExecutionProvider::GetCapability, "
        "number of partitions supported by CoreML: 5 "
        "number of nodes in the graph: 320 "
        "number of nodes supported by CoreML: 314"
    )
    parsed = coreml.parse_coreml_capability(log)

    assert parsed["partitions"] == 5
    assert parsed["graph_nodes"] == 320
    assert parsed["supported_nodes"] == 314
    assert parsed["unsupported_nodes"] == 6


def test_profile_counts_each_kernel_once():
    event = {
        "cat": "Node",
        "name": "/model.23/Concat_kernel_time",
        "args": {"provider": "CPUExecutionProvider", "op_name": "Concat"},
    }
    core = {
        "cat": "Node",
        "name": "fused_CoreML_kernel_time",
        "args": {"provider": "CoreMLExecutionProvider", "op_name": "CoreML"},
    }
    summary = coreml.summarize_ort_profile([event, event, core, core, {"cat": "Session"}])

    assert summary["cpu_kernels"] == 1
    assert summary["coreml_kernels"] == 1
    assert summary["cpu_operators"] == {"Concat": 1}


def test_renumbered_tracks_stay_mapped():
    reference = [
        _row(1, [_box(0, 0, 10, 10, cls=2, track_id=1), _box(20, 20, 30, 30, cls=2, track_id=2)]),
        _row(2, [_box(0, 0, 10, 10, cls=2, track_id=1)]),
    ]
    candidate = [
        _row(1, [_box(0, 0, 10, 10, cls=2, track_id=9), _box(20, 20, 30, 30, cls=2, track_id=8)]),
        _row(2, [_box(0, 0, 10, 10, cls=2, track_id=9)]),
    ]
    mapped = coreml.match_trajectories(reference, candidate)

    assert mapped["mapped"] == 2
    assert mapped["renumbered"] == 2
    assert mapped["same_id"] == 0
    assert mapped["fragmented_reference_tracks"] == 0
    assert mapped["identity_switches"] == 0
    assert mapped["unmatched_reference"] == 0


def test_a_split_track_is_fragmented():
    reference = [
        _row(1, [_box(0, 0, 10, 10, cls=2, track_id=1)]),
        _row(2, [_box(0, 0, 10, 10, cls=2, track_id=1)]),
        _row(3, [_box(0, 0, 10, 10, cls=2, track_id=1)]),
    ]
    candidate = [
        _row(1, [_box(0, 0, 10, 10, cls=2, track_id=4)]),
        _row(2, [_box(0, 0, 10, 10, cls=2, track_id=4)]),
        _row(3, [_box(0, 0, 10, 10, cls=2, track_id=5)]),
    ]
    mapped = coreml.match_trajectories(reference, candidate)

    assert mapped["fragmented_reference_tracks"] == 1
    assert mapped["mapped"] == 1
    assert mapped["unmatched_candidate"] == 1
    assert mapped["identity_switches"] == 1
    assert mapped["pairs"][0]["candidate_id"] == 4


def test_episode_windows_match_without_the_same_track_id():
    reference = [{"track_id": "10", "start_frame": "81", "end_frame": "83"}]
    candidate = [{"track_id": "8", "start_frame": "81", "end_frame": "83"}]
    matched = coreml.match_episodes(reference, candidate)

    assert matched["mapped"] == 1
    assert matched["same_bounds"] == 1
    assert matched["pairs"][0]["reference_track_id"] == "10"
    assert matched["pairs"][0]["candidate_track_id"] == "8"


def test_forced_provider_is_restored_when_the_run_raises(monkeypatch):
    class Package:
        class InferenceSession:
            pass

    package = Package()
    original = package.InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", package)

    with pytest.raises(RuntimeError):
        with coreml._forced_sessions([("CoreMLExecutionProvider", {})]):
            assert package.InferenceSession is not original
            raise RuntimeError("stop")

    assert package.InferenceSession is original


def test_missing_onnx_runtime_exits_before_a_video(monkeypatch, capsys):
    module = coreml._onnx_benchmark()
    monkeypatch.setattr(module, "probe_onnx_runtime", lambda: {"available": False})

    code = coreml.main([])

    assert code == 2
    assert "not installed" in capsys.readouterr().err


def _row(frame: int, boxes: list[dict]) -> dict:
    return {"frame": frame, "boxes": boxes}


def _box(x1, y1, x2, y2, *, cls: int, track_id: int) -> dict:
    return {"cls": cls, "conf": 0.9, "xyxy": (x1, y1, x2, y2), "track_id": track_id}
