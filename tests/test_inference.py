"""Optional Core ML backend selection without ONNX Runtime."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from roadvision.cli import main
from roadvision.config import ConfigError, RoadVisionConfig, load_config, validate_config
from roadvision.inference import (
    COREML_PROVIDER_OPTIONS,
    SessionProviders,
    invalid_provider_options,
    require_coreml_runtime,
)
from roadvision.pipeline import PipelineError, _write_metadata, run_video


ROOT = Path(__file__).resolve().parents[1]


def test_default_backend_is_pytorch():
    config = load_config()
    checked_in = load_config(ROOT / "configs" / "roadvision.yaml")

    assert config.detector.backend == "pytorch"
    assert checked_in.detector.backend == "pytorch"
    assert config.detector.model_path == "yolo11n.pt"
    assert config.detector.device == "mps"


def test_existing_cli_command_keeps_pytorch(monkeypatch, tmp_path, capsys):
    calls = []

    def fake(config, profile=False):
        calls.append(config)
        raise PipelineError("stop before a video")

    monkeypatch.setattr("roadvision.cli.run_video", fake)
    status = main([
        "run",
        "--input",
        "data/videos/test_drive.mp4",
        "--output",
        str(tmp_path / "out"),
        "--device",
        "cpu",
        "--max-frames",
        "1",
    ])

    assert status == 1
    assert calls[0].detector.backend == "pytorch"
    assert calls[0].detector.model_path == "yolo11n.pt"
    assert main(["run", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--backend" in help_text
    assert "--model" in help_text


def test_coreml_selection_requires_cpu_and_an_onnx_model(tmp_path):
    model = tmp_path / "yolo11n.onnx"
    model.write_bytes(b"not-a-real-model")
    base = RoadVisionConfig()
    selected = replace(
        base,
        detector=replace(
            base.detector,
            backend="onnx-coreml",
            device="cpu",
            model_path=str(model),
        ),
    )
    validate_config(selected)

    with pytest.raises(ConfigError, match="requires detector.device"):
        validate_config(
            replace(selected, detector=replace(selected.detector, device="mps"))
        )
    with pytest.raises(ConfigError, match="requires an .onnx model"):
        validate_config(
            replace(
                selected,
                detector=replace(selected.detector, model_path=str(tmp_path / "yolo11n.pt")),
            )
        )


def test_cli_rejects_an_unknown_backend():
    assert main(["run", "--backend", "tensorrt"]) == 2


def test_missing_onnxruntime_is_reported(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    def missing():
        raise ImportError("onnxruntime is absent")

    monkeypatch.setattr("roadvision.inference._import_onnxruntime", missing)

    with pytest.raises(PipelineError, match="does not fall back to PyTorch"):
        require_coreml_runtime()


def test_coreml_is_rejected_away_from_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(PipelineError, match="macOS"):
        require_coreml_runtime()


def test_coreml_provider_must_be_installed(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    class Runtime:
        __version__ = "0"

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        "roadvision.inference._import_onnxruntime",
        lambda: Runtime(),
    )

    with pytest.raises(PipelineError, match="CoreMLExecutionProvider is not available"):
        require_coreml_runtime()


def test_invalid_provider_options_are_named():
    assert invalid_provider_options(COREML_PROVIDER_OPTIONS) == []
    assert "MLComputeUnits" in invalid_provider_options(
        {**COREML_PROVIDER_OPTIONS, "MLComputeUnits": "TPU"}
    )
    assert "ModelFormat" in invalid_provider_options(
        {**COREML_PROVIDER_OPTIONS, "ModelFormat": "FP16"}
    )


def test_provider_override_is_restored_after_an_exception(monkeypatch):
    class Package:
        class InferenceSession:
            pass

    package = Package()
    original = package.InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", package)

    with pytest.raises(RuntimeError, match="stop"):
        with SessionProviders([("CoreMLExecutionProvider", {}), "CPUExecutionProvider"]):
            assert package.InferenceSession is not original
            raise RuntimeError("stop")

    assert package.InferenceSession is original


def test_coreml_runs_do_not_leak_the_provider_override(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_clip(video)
    model = tmp_path / "yolo11n.onnx"
    model.write_bytes(b"onnx-bytes")

    class Package:
        __version__ = "1.32.0"

        class InferenceSession:
            pass

    package = Package()
    original = package.InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", package)
    monkeypatch.setattr("roadvision.inference.require_coreml_runtime", lambda: None)
    monkeypatch.setattr("roadvision.inference.validate_onnx_model", lambda path: None)

    class Session:
        def get_providers(self):
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CoreMLExecutionProvider": dict(COREML_PROVIDER_OPTIONS)}

    class FakeYOLO:
        created = []

        def __init__(self, path):
            self.path = path
            self.predictor = type("Predictor", (), {"model": type("Backend", (), {"session": Session()})()})()
            FakeYOLO.created.append(self)

        def track(self, source, **kwargs):
            assert package.InferenceSession is not original
            return [type("Result", (), {"boxes": None})()]

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    config = _coreml_config(tmp_path, video, model, "run-a")
    first = run_video(config)
    assert package.InferenceSession is original
    second = run_video(replace(
        config,
        video=replace(config.video, output_dir=str(tmp_path / "run-b")),
    ))

    assert package.InferenceSession is original
    assert len(FakeYOLO.created) == 2
    assert FakeYOLO.created[0] is not FakeYOLO.created[1]
    document = json.loads(first.metadata_json.read_text(encoding="utf-8"))
    assert document["backend"]["name"] == "onnx-coreml"
    assert document["backend"]["experimental"] is True
    assert document["backend"]["execution_providers"][0] == "CoreMLExecutionProvider"
    assert document["backend"]["provider_options"]["ModelFormat"] == "MLProgram"
    assert document["backend"]["model_sha256"]
    assert "did not profile individual graph nodes" in document["backend"]["node_assignment"]
    assert json.loads(second.metadata_json.read_text(encoding="utf-8"))["backend"]["name"] == "onnx-coreml"


def test_concurrent_coreml_initialization_is_rejected(monkeypatch):
    class Package:
        class InferenceSession:
            pass

    package = Package()
    original = package.InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", package)
    outer = SessionProviders([("CoreMLExecutionProvider", {}), "CPUExecutionProvider"])
    outer.__enter__()
    try:
        with pytest.raises(PipelineError, match="concurrent"):
            SessionProviders([("CoreMLExecutionProvider", {})]).__enter__()
        assert package.InferenceSession is not original
    finally:
        outer.__exit__(None, None, None)

    assert package.InferenceSession is original


def test_pytorch_initialization_leaves_the_onnx_constructor_alone(monkeypatch):
    class Package:
        class InferenceSession:
            pass

    package = Package()
    original = package.InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", package)

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    from roadvision.pipeline import YoloByteTrack

    YoloByteTrack(RoadVisionConfig())
    assert package.InferenceSession is original


def test_incompatible_onnx_models_are_rejected(monkeypatch, tmp_path):
    class Runtime:
        class InferenceSession:
            def __init__(self, path, sess_options=None, providers=None, **kwargs):
                self.path = str(path)
                if self.path.endswith("bad.onnx"):
                    raise RuntimeError("invalid protobuf")

            def get_inputs(self):
                shape = [1, 3, "h", "w"] if self.path.endswith("dynamic.onnx") else [1, 3, 640, 640]
                return [type("Input", (), {"shape": shape})()]

            def get_outputs(self):
                shape = [1, 300, 6] if self.path.endswith("nms.onnx") else [1, 84, 8400]
                return [type("Output", (), {"shape": shape})()]

    monkeypatch.setattr("roadvision.inference._import_onnxruntime", lambda: Runtime)
    from roadvision.inference import validate_onnx_model

    validate_onnx_model(tmp_path / "yolo11n.onnx")
    with pytest.raises(PipelineError, match="invalid protobuf"):
        validate_onnx_model(tmp_path / "bad.onnx")
    with pytest.raises(PipelineError, match="static YOLO11n"):
        validate_onnx_model(tmp_path / "dynamic.onnx")
    with pytest.raises(PipelineError, match="NMS must stay outside"):
        validate_onnx_model(tmp_path / "nms.onnx")


def test_default_metadata_has_no_experimental_backend_record(tmp_path):
    config = RoadVisionConfig()
    path = tmp_path / "metadata.json"
    _write_metadata(
        path,
        config,
        fps=30.0,
        width=16,
        height=16,
        container_frames=1,
        frames_processed=1,
        observation_count=0,
        episode_count=0,
        accepted_lane_frames=0,
        elapsed_seconds=0.1,
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == "1"
    assert document["configuration"]["detector"]["backend"] == "pytorch"
    assert "backend" not in document
    assert document["outputs"]["observation_columns"]


def _coreml_config(tmp_path: Path, video: Path, model: Path, folder: str) -> RoadVisionConfig:
    base = RoadVisionConfig()
    return replace(
        base,
        video=replace(
            base.video,
            input_path=str(video),
            output_dir=str(tmp_path / folder),
            max_frames=1,
        ),
        detector=replace(
            base.detector,
            backend="onnx-coreml",
            device="cpu",
            model_path=str(model),
        ),
        tracker=replace(
            base.tracker,
            config_path=str(ROOT / "configs" / "bytetrack_threshold.yaml"),
        ),
    )


def _write_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
    if not writer.isOpened():
        raise RuntimeError("Could not create the synthetic test video")
    try:
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()
