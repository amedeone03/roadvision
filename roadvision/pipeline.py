"""End-to-end RoadVision video run.

The frame loop follows ``scripts/run_integrated_video.py``:
lane detection and temporal smoothing run on every frame, then
YOLO and ByteTrack, then conservative lane association. Each
tracked box is passed to a new ``TrackMotionProcessor``.

Observations are written as they are produced. A new model,
lane filter, smoother, and track processor are created for
every call.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from roadvision.annotate import VehicleOverlay, annotate_frame
from roadvision.config import (
    OBSERVATION_COLUMNS,
    CSV_TIMESTAMP_DECIMALS,
    ConfigError,
    RoadVisionConfig,
    to_mapping,
    validate_runtime_paths,
)
from roadvision.profiling import FrameProfiler
from roadvision.tracks import TrackMotionProcessor
from src.association.vehicle_lane import (
    build_lane_polygon,
    classify_vehicle_lane_conservative,
    classify_vehicle_lane_detailed,
    get_bottom_center,
)
from src.lanes.lane_detector import detect_lane
from src.lanes.lane_smoother import LaneSmoother
from src.lanes.temporal_filter import TemporalLaneFilter


logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

EVENT_COLUMNS = (
    "track_id",
    "start_frame",
    "end_frame",
    "confirmation_frame",
    "duration_frames",
    "duration_s",
    "positive_frames",
    "interruption_frames",
    "min_tau_proxy_s",
)


class PipelineError(RuntimeError):
    """The video run could not be completed."""


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """One ByteTrack observation for the current frame."""

    bbox: tuple[float, float, float, float]
    track_id: int
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Paths and counts for one completed run."""

    annotated_video: Path
    observations_csv: Path
    events_csv: Path
    metadata_json: Path
    frames_processed: int
    observation_count: int
    episode_count: int
    elapsed_seconds: float
    frame_timings_csv: Path | None = None
    profiling_summary_json: Path | None = None


class YoloByteTrack:
    """One YOLO model and its ByteTrack state for a single run."""

    def __init__(self, config: RoadVisionConfig) -> None:
        from ultralytics import YOLO

        self._config = config
        self._model = YOLO(config.detector.model_path)

    def track(self, frame) -> list[TrackedObject]:
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._config.tracker.config_path,
            conf=self._config.detector.confidence,
            classes=list(self._config.detector.classes),
            device=self._config.detector.device,
            verbose=False,
        )
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return []

        coords = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        objects = []

        for bbox, track_id, class_id, confidence in zip(
            coords,
            track_ids,
            class_ids,
            confidences,
        ):
            class_id = int(class_id)
            objects.append(
                TrackedObject(
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    track_id=int(track_id),
                    class_id=class_id,
                    class_name=str(self._model.names[class_id]),
                    confidence=float(confidence),
                )
            )

        return objects


def require_device(device: str) -> None:
    """Reject a requested accelerator that this machine cannot use.

    CPU is always accepted. ``mps`` and ``cuda`` are not replaced
    with CPU when they are unavailable.
    """

    if device == "cpu":
        return

    import torch

    if device == "mps":
        mps = getattr(torch.backends, "mps", None)

        if mps is None or not mps.is_available():
            raise PipelineError(
                "Requested device 'mps' is not available. "
                "RoadVision does not fall back to CPU."
            )

        return

    if device == "cuda":
        if not torch.cuda.is_available():
            raise PipelineError(
                "Requested device 'cuda' is not available. "
                "RoadVision does not fall back to CPU."
            )

        return

    raise PipelineError(
        f"Unsupported device {device!r}. Expected cpu, mps, or cuda."
    )


def run_video(
    config: RoadVisionConfig,
    *,
    profile: bool = False,
) -> PipelineResult:
    """Process ``config.video.input_path`` and write the four outputs.

    ``config.video.max_frames`` stops the run early when it is set.
    The measured container frame rate is passed to the track processor.
    ``profile`` writes frame timings beside those outputs and
    synchronizes MPS or CUDA after inference. The default does neither.
    """

    if (
        config.video.max_frames is not None
        and config.video.max_frames < 1
    ):
        raise ConfigError("video.max_frames must be >= 1 or null")

    require_device(config.detector.device)
    validate_runtime_paths(config)

    output_dir = Path(config.video.output_dir)
    annotated_path = output_dir / config.output.annotated_video
    observations_path = output_dir / config.output.observations_csv
    events_path = output_dir / config.output.events_csv
    metadata_path = output_dir / config.output.metadata_json

    cap = None
    writer = None
    observation_file = None
    started = time.perf_counter()

    try:
        cap = cv2.VideoCapture(str(config.video.input_path))

        if not cap.isOpened():
            raise PipelineError(
                f"Cannot open video: {config.video.input_path}"
            )

        fps, width, height, container_frames = _video_properties(cap)
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise PipelineError(
                f"Cannot create video: {annotated_path}. "
                "The mp4v writer did not open."
            )

        detector = YoloByteTrack(config)
        lane_filter = TemporalLaneFilter(
            max_jump=config.lane_filter.max_jump,
            max_reference_age=config.lane_filter.max_reference_age,
            max_candidate_gap=config.lane_filter.max_candidate_gap,
        )
        lane_smoother = LaneSmoother(alpha=config.lane_smoother.alpha)
        processor = TrackMotionProcessor(fps, config)
        profiler = FrameProfiler(
            enabled=profile,
            device=config.detector.device,
        )

        observation_file = observations_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        observation_writer = csv.writer(observation_file)
        observation_writer.writerow(OBSERVATION_COLUMNS)

        logger.info(
            "Processing %s on %s at %.5f FPS",
            config.video.input_path,
            config.detector.device,
            fps,
        )

        frame_number = 0
        observation_count = 0
        accepted_lane_frames = 0

        while True:
            if (
                config.video.max_frames is not None
                and frame_number >= config.video.max_frames
            ):
                break

            profiler.begin_frame()

            try:
                with profiler.span("decode_ms"):
                    ok, frame = cap.read()

                if not ok:
                    profiler.discard_frame()
                    break

                if frame.shape[1] != width or frame.shape[0] != height:
                    raise PipelineError(
                        "A decoded frame does not match the video size "
                        f"({width}x{height})."
                    )

                frame_number += 1

                with profiler.span("lane_detection_ms"):
                    _, lane_info = detect_lane(frame)

                geometry_valid = bool(lane_info["valid"])

                with profiler.span("lane_temporal_ms"):
                    temporal_accepted, temporal_reason = lane_filter.update(
                        frame_number,
                        lane_info,
                    )

                    if temporal_accepted:
                        accepted_lane_frames += 1

                    smoothed_left, smoothed_right = lane_smoother.update(
                        lane_info,
                        temporal_accepted,
                        temporal_reason,
                    )
                    lane_left = smoothed_left if temporal_accepted else None
                    lane_right = smoothed_right if temporal_accepted else None
                    lane_polygon = None

                    if (
                        temporal_accepted
                        and smoothed_left is not None
                        and smoothed_right is not None
                    ):
                        lane_polygon = build_lane_polygon(
                            smoothed_left,
                            smoothed_right,
                        )

                with profiler.span("inference_tracking_ms"):
                    tracked_objects = detector.track(frame)
                    profiler.synchronize()

                vehicles: list[VehicleOverlay] = []

                for tracked in tracked_objects:
                    with profiler.span("lane_association_ms"):
                        baseline_status, baseline_reason = (
                            classify_vehicle_lane_detailed(
                                tracked.bbox,
                                lane_polygon,
                            )
                        )
                        lane_status, lane_reason, margin = (
                            classify_vehicle_lane_conservative(
                                tracked.bbox,
                                lane_left,
                                lane_right,
                                margin=config.association.lateral_margin,
                            )
                        )

                    with profiler.span("motion_event_ms"):
                        estimate = processor.update(
                            frame_number,
                            tracked.track_id,
                            tracked.bbox,
                            lane_status,
                        )

                    with profiler.span("csv_write_ms"):
                        observation_writer.writerow(
                            _observation_row(
                                frame_number=frame_number,
                                fps=fps,
                                tracked=tracked,
                                lane_status=lane_status,
                                lane_reason=lane_reason,
                                baseline_status=baseline_status,
                                baseline_reason=baseline_reason,
                                margin=margin,
                                temporal_accepted=temporal_accepted,
                                geometry_valid=geometry_valid,
                                temporal_reason=temporal_reason,
                                estimate=estimate,
                            )
                        )

                    observation_count += 1
                    vehicles.append(
                        VehicleOverlay(
                            bbox=tracked.bbox,
                            track_id=tracked.track_id,
                            class_name=tracked.class_name,
                            lane_status=lane_status,
                            lane_reason=lane_reason,
                            approach_state=estimate.approach_state,
                            event_state=estimate.event_state,
                            tau_proxy_s=estimate.tau_proxy_s,
                        )
                    )

                with profiler.span("annotation_ms"):
                    annotated = annotate_frame(
                        frame,
                        lane_left,
                        lane_right,
                        vehicles,
                        frame_number,
                        temporal_accepted,
                    )

                with profiler.span("video_write_ms"):
                    writer.write(annotated)

                with profiler.span("csv_write_ms"):
                    observation_file.flush()

                profiler.end_frame(
                    frame=frame_number,
                    tracked_objects=len(tracked_objects),
                    lane_association_active=bool(tracked_objects),
                )
            except Exception:
                profiler.discard_frame()
                raise

            if frame_number % 100 == 0:
                logger.info("Processed %s frames", frame_number)

        episodes = processor.finish()
        _write_events(events_path, episodes)
        elapsed = time.perf_counter() - started
        timings_path = None
        summary_path = None

        if profiler.enabled:
            timings_path, summary_path = profiler.write(
                output_dir,
                {
                    "model_path": config.detector.model_path,
                    "confidence": config.detector.confidence,
                    "classes": list(config.detector.classes),
                    "tracker_path": config.tracker.config_path,
                    "video_path": config.video.input_path,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frames_processed": frame_number,
                    "pipeline_elapsed_seconds": elapsed,
                },
            )

        _write_metadata(
            metadata_path,
            config=config,
            fps=fps,
            width=width,
            height=height,
            container_frames=container_frames,
            frames_processed=frame_number,
            observation_count=observation_count,
            episode_count=len(episodes),
            accepted_lane_frames=accepted_lane_frames,
            elapsed_seconds=elapsed,
        )
        logger.info(
            "Wrote %s frames, %s observations, %s episodes to %s",
            frame_number,
            observation_count,
            len(episodes),
            output_dir,
        )
        return PipelineResult(
            annotated_video=annotated_path,
            observations_csv=observations_path,
            events_csv=events_path,
            metadata_json=metadata_path,
            frames_processed=frame_number,
            observation_count=observation_count,
            episode_count=len(episodes),
            elapsed_seconds=elapsed,
            frame_timings_csv=timings_path,
            profiling_summary_json=summary_path,
        )
    finally:
        if observation_file is not None:
            observation_file.close()

        if writer is not None:
            writer.release()

        if cap is not None:
            cap.release()


def _video_properties(cap) -> tuple[float, int, int, int | None]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not math.isfinite(fps) or fps <= 0.0:
        raise PipelineError(
            f"Invalid input video FPS: {fps}. "
            "The measured frame rate is required for motion timing."
        )

    if width <= 0 or height <= 0:
        raise PipelineError(
            f"Invalid input video size: {width}x{height}."
        )

    container_frames = (
        int(raw_count)
        if math.isfinite(raw_count) and raw_count >= 0.0
        else None
    )
    return fps, width, height, container_frames


def _observation_row(
    frame_number: int,
    fps: float,
    tracked: TrackedObject,
    lane_status: str,
    lane_reason: str,
    baseline_status: str,
    baseline_reason: str,
    margin,
    temporal_accepted: bool,
    geometry_valid: bool,
    temporal_reason: str,
    estimate,
) -> list:
    x1, y1, x2, y2 = tracked.bbox
    contact_x, contact_y = get_bottom_center(tracked.bbox)
    timestamp = (frame_number - 1) / fps
    values = {
        "frame": frame_number,
        "timestamp": round(timestamp, CSV_TIMESTAMP_DECIMALS),
        "track_id": tracked.track_id,
        "class_id": tracked.class_id,
        "class_name": tracked.class_name,
        "confidence": round(tracked.confidence, 4),
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "contact_x": round(contact_x, 2),
        "contact_y": round(contact_y, 2),
        "lane_status": lane_status,
        "lane_reason": lane_reason,
        "baseline_status": baseline_status,
        "baseline_reason": baseline_reason,
        "normalized_lateral_margin": (
            "" if margin is None else round(float(margin), 5)
        ),
        "lane_valid": temporal_accepted,
        "lane_geometry_valid": geometry_valid,
        "lane_temporal_reason": temporal_reason,
        "height": estimate.height,
        "time_seconds": estimate.time_seconds,
        "growth_robust_5": _missing(estimate.growth_robust_5),
        "approach_state": estimate.approach_state,
        "tau_proxy_s": _missing(estimate.tau_proxy_s),
        "evidence": estimate.evidence,
        "event_state": estimate.event_state,
    }
    return [values[column] for column in OBSERVATION_COLUMNS]


def _missing(value):
    if value is None:
        return ""

    if isinstance(value, float) and not math.isfinite(value):
        return ""

    return value


def _write_events(path: Path, episodes) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVENT_COLUMNS)

        for episode in episodes:
            writer.writerow([
                episode.track_id,
                episode.start_frame,
                episode.end_frame,
                episode.confirmation_frame,
                episode.duration_frames,
                episode.duration_s,
                episode.positive_frames,
                episode.interruption_frames,
                _missing(episode.min_tau_proxy_s),
            ])


def _write_metadata(
    path: Path,
    config: RoadVisionConfig,
    fps: float,
    width: int,
    height: int,
    container_frames: int | None,
    frames_processed: int,
    observation_count: int,
    episode_count: int,
    accepted_lane_frames: int,
    elapsed_seconds: float,
) -> None:
    tracker_settings = _read_tracker_settings(config.tracker.config_path)
    throughput = (
        frames_processed / elapsed_seconds
        if elapsed_seconds > 0.0
        else None
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "configuration": to_mapping(config),
        "model": {
            "path": config.detector.model_path,
            "confidence": config.detector.confidence,
            "classes": list(config.detector.classes),
            "device": config.detector.device,
        },
        "tracker": {
            "path": config.tracker.config_path,
            "settings": tracker_settings,
        },
        "video": {
            "path": config.video.input_path,
            "width": width,
            "height": height,
            "fps": fps,
            "container_frame_count": container_frames,
            "frames_processed": frames_processed,
        },
        "outputs": {
            "annotated_video": config.output.annotated_video,
            "observations_csv": config.output.observations_csv,
            "events_csv": config.output.events_csv,
            "metadata_json": config.output.metadata_json,
            "observation_columns": list(OBSERVATION_COLUMNS),
            "event_columns": list(EVENT_COLUMNS),
        },
        "formatting": {
            "timestamp_decimals": CSV_TIMESTAMP_DECIMALS,
            "bbox_decimals": 2,
            "confidence_decimals": 4,
            "lateral_margin_decimals": 5,
            "motion_uses_unrounded_detector_boxes": True,
        },
        "counts": {
            "observations": observation_count,
            "episodes": episode_count,
            "accepted_lane_frames": accepted_lane_frames,
        },
        "timing": {
            "elapsed_seconds": round(elapsed_seconds, 6),
            "frames_per_second": (
                None if throughput is None else round(throughput, 6)
            ),
        },
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_tracker_settings(path: str):
    import yaml

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(loaded, dict):
        raise PipelineError(f"Tracker YAML must be a mapping: {path}")

    return loaded
