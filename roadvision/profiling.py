"""Optional per-frame wall-clock profiling for one video run.

Timers use ``time.perf_counter_ns``. A disabled profiler does not
read the clock and does not synchronize the GPU.

``inference_tracking_ms`` is the combined Ultralytics ``track()``
call. YOLO and ByteTrack are not separated. On MPS or CUDA the
profiler synchronizes the device before that stage timer stops, so
the value is synchronized wall-clock time, not a GPU kernel
duration. Synchronization is skipped when profiling is off.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


FRAME_TIMINGS_NAME = "frame_timings.csv"
PROFILING_SUMMARY_NAME = "profiling_summary.json"
WARMUP_FRAMES = 1
PERCENTILE_MINIMUM_FRAMES = 30

STAGE_NAMES = (
    "decode_ms",
    "lane_detection_ms",
    "lane_temporal_ms",
    "inference_tracking_ms",
    "lane_association_ms",
    "motion_event_ms",
    "annotation_ms",
    "video_write_ms",
    "csv_write_ms",
)

FRAME_COLUMNS = (
    "frame",
    "warmup",
    "tracked_objects",
    "lane_association_active",
    *STAGE_NAMES,
    "other_ms",
    "frame_total_ms",
)

_NS_PER_MS = 1_000_000


class _InactiveSpan:
    """Context manager used when profiling is off."""

    def __enter__(self) -> _InactiveSpan:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


_INACTIVE = _InactiveSpan()


@dataclass
class _Span:
    """Accumulate wall-clock time for one named stage."""

    profiler: FrameProfiler
    name: str
    _started: int = 0

    def __enter__(self) -> _Span:
        self._started = time.perf_counter_ns()
        return self

    def __exit__(self, *args: object) -> bool:
        self.profiler._add(self.name, time.perf_counter_ns() - self._started)
        return False


@dataclass
class FrameProfiler:
    """Collect exclusive stage times for each completed frame."""

    enabled: bool
    device: str
    frames: list[dict[str, object]] = field(default_factory=list)
    synchronization_calls: int = 0
    _open: dict[str, int] | None = None

    def span(self, name: str) -> _Span | _InactiveSpan:
        """Return a timer for ``name``, or a no-op when disabled."""

        if not self.enabled:
            return _INACTIVE

        if name not in STAGE_NAMES:
            raise KeyError(f"Unknown profiling stage {name!r}")

        return _Span(self, name)

    def begin_frame(self) -> None:
        """Start the outer clock for the frame about to be processed."""

        if not self.enabled:
            return

        self._open = {name: 0 for name in STAGE_NAMES}
        self._open["_origin"] = time.perf_counter_ns()

    def discard_frame(self) -> None:
        """Drop an unfinished frame after a failed read or an error."""

        self._open = None

    def synchronize(self) -> None:
        """Wait for the selected accelerator. CPU and disabled runs do not."""

        if not self.enabled:
            return

        if synchronize_device(self.device):
            self.synchronization_calls += 1

    def end_frame(
        self,
        frame: int,
        tracked_objects: int,
        lane_association_active: bool,
    ) -> None:
        """Store one completed frame. The first stored frame is warmup."""

        if not self.enabled or self._open is None:
            return

        total_ns = time.perf_counter_ns() - self._open["_origin"]
        stage_ns = {name: self._open[name] for name in STAGE_NAMES}
        other_ns = total_ns - sum(stage_ns.values())
        record: dict[str, object] = {
            "frame": frame,
            "warmup": len(self.frames) < WARMUP_FRAMES,
            "tracked_objects": tracked_objects,
            "lane_association_active": lane_association_active,
            "other_ms": other_ns / _NS_PER_MS,
            "frame_total_ms": total_ns / _NS_PER_MS,
        }

        for name, elapsed in stage_ns.items():
            record[name] = elapsed / _NS_PER_MS

        self.frames.append(record)
        self._open = None

    def _add(self, name: str, elapsed_ns: int) -> None:
        if self._open is None:
            return

        self._open[name] += elapsed_ns

    def write(self, output_dir: Path, context: dict[str, object]) -> tuple[Path, Path]:
        """Write the frame CSV and the summary JSON."""

        output_dir.mkdir(parents=True, exist_ok=True)
        timings = output_dir / FRAME_TIMINGS_NAME
        summary_path = output_dir / PROFILING_SUMMARY_NAME
        _write_timings(timings, self.frames)
        summary_path.write_text(
            json.dumps(
                build_summary(self, context),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return timings, summary_path


def synchronize_device(device: str) -> bool:
    """Block until queued MPS or CUDA work has finished.

    Returns whether a device synchronization call was made.
    """

    if device == "cpu":
        return False

    import torch

    if device == "mps":
        torch.mps.synchronize()
        return True

    if device == "cuda":
        torch.cuda.synchronize()
        return True

    return False


def build_summary(profiler: FrameProfiler, context: dict[str, object]) -> dict:
    """Summarize measured frames. Warmup frames stay in the CSV only."""

    measured = [frame for frame in profiler.frames if not frame["warmup"]]
    sufficient = len(measured) >= PERCENTILE_MINIMUM_FRAMES
    stages = {
        name: _stage_summary([float(frame[name]) for frame in measured])
        for name in (*STAGE_NAMES, "other_ms", "frame_total_ms")
    }
    elapsed = context.get("pipeline_elapsed_seconds")
    frames_processed = int(context.get("frames_processed") or 0)
    throughput = None

    if isinstance(elapsed, (int, float)) and elapsed > 0.0:
        throughput = frames_processed / float(elapsed)

    return {
        "schema_version": "1",
        "environment": _environment(),
        "run": {
            "device": profiler.device,
            "gpu_synchronization_enabled": profiler.device in {"mps", "cuda"},
            "gpu_synchronization_calls": profiler.synchronization_calls,
            "model_path": context.get("model_path"),
            "confidence": context.get("confidence"),
            "classes": context.get("classes"),
            "tracker_path": context.get("tracker_path"),
            "video_path": context.get("video_path"),
            "width": context.get("width"),
            "height": context.get("height"),
            "fps": context.get("fps"),
            "frames_processed": frames_processed,
            "pipeline_elapsed_seconds": _round(elapsed),
            "pipeline_frames_per_second": _round(throughput),
            "recorded_frames": len(profiler.frames),
            "warmup_frames": sum(
                1 for frame in profiler.frames if frame["warmup"]
            ),
            "measured_frames": len(measured),
            "percentile_minimum_frames": PERCENTILE_MINIMUM_FRAMES,
            "percentile_sample_sufficient": sufficient,
            "lane_association_active_frames": sum(
                1
                for frame in measured
                if frame["lane_association_active"]
            ),
        },
        "stages": stages,
        "timing_definition": {
            "clock": "time.perf_counter_ns wall clock",
            "stage_times_are_exclusive": True,
            "frame_total_ms": (
                "Wall clock from the start of frame decoding through "
                "the observation-file flush. other_ms is the residual "
                "inside that interval, including the frame-size check "
                "and overlay assembly."
            ),
            "gpu_stage": "inference_tracking_ms",
            "gpu_measurement": (
                "Combined Ultralytics track() wall clock, including "
                "copying boxes back to the CPU. MPS and CUDA runs "
                "synchronize before this timer stops. The value is not "
                "a GPU kernel duration, and YOLO is not split from "
                "ByteTrack."
            ),
            "synchronization_can_reduce_throughput": True,
            "warmup_policy": (
                "The first completed frame is stored in "
                "frame_timings.csv and excluded from stage statistics "
                "and exported stage totals."
            ),
            "pipeline_throughput": (
                "pipeline_frames_per_second divides processed frames by "
                "the clock from process start through events.csv. That "
                "interval includes model construction, video reader and "
                "writer setup, the frame loop, episode finalization, "
                "and events.csv. metadata.json and the profiling files "
                "are written after the clock stops. The rate is not "
                "the sum of the stage times."
            ),
            "outside_frame_stages": [
                "model construction",
                "video reader and writer setup",
                "episode finalization",
                "events.csv",
            ],
            "after_pipeline_clock": [
                "metadata.json",
                "frame_timings.csv",
                "profiling_summary.json",
            ],
        },
    }


def _stage_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "total_ms": None,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "total_ms": _round(float(array.sum())),
        "mean_ms": _round(float(array.mean())),
        "median_ms": _round(float(np.percentile(array, 50, method="linear"))),
        "p95_ms": _round(float(np.percentile(array, 95, method="linear"))),
        "p99_ms": _round(float(np.percentile(array, 99, method="linear"))),
        "min_ms": _round(float(array.min())),
        "max_ms": _round(float(array.max())),
    }


def _environment() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "cpu_architecture": platform.machine(),
        "torch": _package_version("torch"),
        "ultralytics": _package_version("ultralytics"),
        "implementation": sys.implementation.name,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _round(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None

    return round(float(value), 6)


def _write_timings(path: Path, frames: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_COLUMNS)
        writer.writeheader()

        for frame in frames:
            row = {}

            for column in FRAME_COLUMNS:
                value = frame[column]

                if isinstance(value, float):
                    row[column] = f"{value:.6f}"
                elif isinstance(value, bool):
                    row[column] = "true" if value else "false"
                else:
                    row[column] = value

            writer.writerow(row)
