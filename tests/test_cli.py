"""Command-line overrides and ordinary error reporting."""

import json
import subprocess
import sys
from pathlib import Path

from roadvision.cli import main
from roadvision.config import RoadVisionConfig, load_config
from roadvision.pipeline import PipelineError, PipelineResult


ROOT = Path(__file__).resolve().parents[1]


def test_module_help_lists_the_run_command():
    completed = subprocess.run(
        [sys.executable, "-m", "roadvision", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "run" in completed.stdout
    assert "unrounded" in completed.stdout
    assert "not a physical collision warning" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_default_configuration_is_passed_unchanged(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    calls = _patch_run(monkeypatch)

    assert main(["run"]) == 0

    assert calls[0] == load_config()
    output = capsys.readouterr().out
    assert "Model: yolo11n.pt" in output
    assert "Device: mps" in output
    assert "Frames processed: 4" in output
    assert "data/videos/test_drive.mp4" in output


def test_cli_overrides_replace_file_values_without_editing_the_file(
    monkeypatch,
    tmp_path,
    capsys,
):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join([
            "video:",
            "  input_path: data/videos/from-file.mp4",
            "  output_dir: runs/from-file",
            "  max_frames: 7",
            "detector:",
            "  device: cpu",
            "  confidence: 0.4",
            "",
        ]),
        encoding="utf-8",
    )
    original = config_path.read_text(encoding="utf-8")
    calls = _patch_run(monkeypatch)

    status = main([
        "run",
        "--config",
        str(config_path),
        "--input",
        "data/videos/from-cli.mp4",
        "--output",
        str(tmp_path / "from-cli"),
        "--device",
        "mps",
        "--max-frames",
        "100",
    ])

    assert status == 0
    assert config_path.read_text(encoding="utf-8") == original
    config = calls[0]
    assert config.video.input_path == "data/videos/from-cli.mp4"
    assert config.video.output_dir == str(tmp_path / "from-cli")
    assert config.video.max_frames == 100
    assert config.detector.device == "mps"
    assert config.detector.confidence == 0.4
    assert config.detector.model_path == RoadVisionConfig().detector.model_path
    output = capsys.readouterr().out
    assert "not a collision warning" in output
    assert "unrounded detector coordinates" in output


def test_invalid_arguments_exit_without_a_traceback(capsys):
    assert main(["run", "--device", "tpu"]) == 2
    assert main(["run", "--max-frames", "0"]) == 2
    assert main(["run", "--max-frames", "1.5"]) == 2
    assert main(["inspect"]) == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "mps" in captured.err
    assert "max-frames must be >= 1" in captured.err


def test_missing_configuration_file_is_reported(capsys):
    status = main(["run", "--config", "configs/missing-roadvision.yaml"])

    assert status == 1
    captured = capsys.readouterr()
    assert "Configuration file not found" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_missing_input_video_is_reported(tmp_path, capsys):
    status = main([
        "run",
        "--input",
        str(tmp_path / "missing.mp4"),
        "--output",
        str(tmp_path / "out"),
        "--device",
        "cpu",
    ])

    assert status == 1
    captured = capsys.readouterr()
    assert "Input video not found" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "out").exists()


def test_unavailable_device_does_not_fall_back(monkeypatch, tmp_path, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status = main([
        "run",
        "--input",
        str(tmp_path / "unused.mp4"),
        "--output",
        str(tmp_path / "out"),
        "--device",
        "cuda",
    ])

    assert status == 1
    captured = capsys.readouterr()
    assert "Requested device 'cuda' is not available" in captured.err
    assert "does not fall back to CPU" in captured.err
    assert "Traceback" not in captured.err


def test_pipeline_errors_are_propagated_without_a_traceback(
    monkeypatch,
    capsys,
):
    def fail(_config):
        raise PipelineError("Cannot create video: annotated.mp4")

    monkeypatch.setattr("roadvision.cli.run_video", fail)

    assert main(["run", "--max-frames", "3"]) == 1
    captured = capsys.readouterr()
    assert "error: Cannot create video: annotated.mp4" in captured.err
    assert "Traceback" not in captured.err


def test_successful_mock_reports_counts_and_outputs(monkeypatch, tmp_path, capsys):
    calls = _patch_run(
        monkeypatch,
        frames=100,
        observations=12,
        episodes=0,
        elapsed=2.5,
    )

    assert main([
        "run",
        "--output",
        str(tmp_path / "smoke"),
        "--max-frames",
        "100",
        "--device",
        "cpu",
    ]) == 0

    assert calls[0].video.max_frames == 100
    assert calls[0].detector.device == "cpu"
    output = capsys.readouterr().out
    assert "Device: cpu" in output
    assert "Frames processed: 100" in output
    assert "Tracked observations: 12" in output
    assert "Completed evidence episodes: 0" in output
    assert "Processing duration: 2.500 s" in output
    assert f"Annotated video: {tmp_path / 'smoke' / 'annotated.mp4'}" in output
    assert f"Observations: {tmp_path / 'smoke' / 'observations.csv'}" in output
    assert f"Events: {tmp_path / 'smoke' / 'events.csv'}" in output
    assert f"Metadata: {tmp_path / 'smoke' / 'metadata.json'}" in output
    assert "1920x1080" in output


def _patch_run(
    monkeypatch,
    *,
    frames=4,
    observations=2,
    episodes=1,
    elapsed=1.25,
):
    calls = []

    def fake(config):
        calls.append(config)
        output_dir = Path(config.video.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = output_dir / "metadata.json"
        metadata.write_text(
            json.dumps({
                "video": {
                    "path": config.video.input_path,
                    "width": 1920,
                    "height": 1080,
                    "fps": 29.97003,
                    "container_frame_count": 1728,
                    "frames_processed": frames,
                }
            }),
            encoding="utf-8",
        )
        return PipelineResult(
            annotated_video=output_dir / "annotated.mp4",
            observations_csv=output_dir / "observations.csv",
            events_csv=output_dir / "events.csv",
            metadata_json=metadata,
            frames_processed=frames,
            observation_count=observations,
            episode_count=episodes,
            elapsed_seconds=elapsed,
        )

    monkeypatch.setattr("roadvision.cli.run_video", fake)
    return calls
