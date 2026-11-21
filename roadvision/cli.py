"""Run one RoadVision video from the terminal.

CLI flags override the loaded configuration in memory. They do
not rewrite the configuration file.

Bounding-box columns in ``observations.csv`` are rounded.
Scale growth is calculated from the detector's unrounded
coordinates. A completed episode is algorithmic image-space
evidence, not a physical collision warning.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from roadvision.config import (
    ConfigError,
    RoadVisionConfig,
    load_config,
    validate_config,
)
from roadvision.pipeline import PipelineError, PipelineResult, run_video


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run one video, and return a process status."""

    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code is None or exc.code == 0:
            return 0

        if isinstance(exc.code, int):
            return exc.code

        print(exc.code, file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        config = load_config(args.config)
        config = apply_overrides(config, args)
        validate_config(config)
        if args.profile:
            result = run_video(config, profile=True)
        else:
            result = run_video(config)
    except (ConfigError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_summary(config, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the ``python -m roadvision`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="roadvision",
        description=(
            "Process one driving video into an annotated video, "
            "observations, and experimental approach-evidence episodes."
        ),
        epilog=(
            "Growth uses the detector's unrounded box coordinates. "
            "Bounding-box columns in observations.csv are rounded. "
            "A completed episode is algorithmic image-space evidence, "
            "not a physical collision warning.\n\n"
            "example:\n"
            "  python -m roadvision run "
            "--input data/videos/test_drive.mp4 "
            "--output runs/smoke --max-frames 100"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="Process one video with the current configuration.",
        description=(
            "Flags replace the corresponding loaded settings for this "
            "process only. Omitted flags keep the configuration-file "
            "or default values."
        ),
    )
    run.add_argument(
        "--input",
        type=Path,
        help="Driving video to process. Overrides video.input_path.",
    )
    run.add_argument(
        "--output",
        type=Path,
        help=(
            "Directory for annotated.mp4, observations.csv, "
            "events.csv, and metadata.json. Overrides video.output_dir."
        ),
    )
    run.add_argument(
        "--config",
        type=Path,
        help="YAML configuration. Omit it to use the built-in defaults.",
    )
    run.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        help=(
            "Inference device. Overrides detector.device. "
            "mps and cuda are not replaced with cpu when unavailable."
        ),
    )
    run.add_argument(
        "--backend",
        choices=("pytorch", "onnx-coreml"),
        help=(
            "Inference backend. pytorch is the default. "
            "onnx-coreml is an experimental Apple Silicon ONNX session "
            "and requires --device cpu and an .onnx --model."
        ),
    )
    run.add_argument(
        "--model",
        type=Path,
        help="Weights or ONNX file. Overrides detector.model_path.",
    )
    run.add_argument(
        "--max-frames",
        type=_positive_frames,
        dest="max_frames",
        help=(
            "Stop after this many frames. Overrides video.max_frames. "
            "Omit it to process the whole video."
        ),
    )
    run.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Write frame_timings.csv and profiling_summary.json. "
            "MPS and CUDA runs synchronize after inference. "
            "Omit it to leave timing and synchronization unchanged."
        ),
    )
    return parser


def apply_overrides(
    config: RoadVisionConfig,
    args: argparse.Namespace,
) -> RoadVisionConfig:
    """Return a copy with the flags that were actually supplied."""

    video = config.video
    detector = config.detector

    if args.input is not None:
        video = replace(video, input_path=str(args.input))

    if args.output is not None:
        video = replace(video, output_dir=str(args.output))

    if args.max_frames is not None:
        video = replace(video, max_frames=args.max_frames)

    if args.device is not None:
        detector = replace(detector, device=args.device)

    if getattr(args, "backend", None) is not None:
        detector = replace(detector, backend=args.backend)

    if getattr(args, "model", None) is not None:
        detector = replace(detector, model_path=str(args.model))

    return replace(config, video=video, detector=detector)


def print_summary(config: RoadVisionConfig, result: PipelineResult) -> None:
    """Print the run that just finished."""

    video = _video_properties(result)
    fps = video.get("fps")
    fps_text = f"{fps:.5f}" if isinstance(fps, (int, float)) else "unknown"
    container_frames = video.get("container_frame_count")
    container_text = (
        str(container_frames) if container_frames is not None else "unknown"
    )
    print(f"Model: {config.detector.model_path}")
    print(f"Backend: {config.detector.backend}")
    print(f"Device: {config.detector.device}")
    print(
        "Input: "
        f"{video.get('path', config.video.input_path)} "
        f"({video.get('width', '?')}x{video.get('height', '?')}, "
        f"{fps_text} fps, {container_text} container frames)"
    )
    print(f"Frames processed: {result.frames_processed}")
    print(f"Tracked observations: {result.observation_count}")
    print(
        "Completed evidence episodes: "
        f"{result.episode_count} "
        "(algorithmic image-space evidence, not a collision warning)"
    )
    print(f"Processing duration: {result.elapsed_seconds:.3f} s")
    print(f"Annotated video: {result.annotated_video}")
    print(f"Observations: {result.observations_csv}")
    print(f"Events: {result.events_csv}")
    print(f"Metadata: {result.metadata_json}")

    if result.frame_timings_csv is not None:
        print(f"Frame timings: {result.frame_timings_csv}")

    if result.profiling_summary_json is not None:
        print(f"Profiling summary: {result.profiling_summary_json}")
    print(
        "Growth is calculated from unrounded detector coordinates. "
        "Bounding boxes in observations.csv are rounded to 2 decimals."
    )


def _video_properties(result: PipelineResult) -> dict:
    if not result.metadata_json.is_file():
        return {}

    document = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    video = document.get("video", {})
    return video if isinstance(video, dict) else {}


def _positive_frames(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"max-frames must be an integer >= 1, got {value!r}"
        ) from exc

    if number < 1:
        raise argparse.ArgumentTypeError("max-frames must be >= 1")

    return number
