"""Time one YOLO.track() call without changing the RoadVision pipeline.

Frames are decoded before the timed section. Each pass builds a new
model and a new tracker. The first completed frame stays out of the
steady-state statistics because Ultralytics warms the model up inside
that call.

``track_call_ms`` is synchronized wall-clock time around ``YOLO.track``
only. ``Results.speed`` is recorded separately and is not synchronized
on MPS. The tracker-callback interval is recorded only when the
installed callback list puts the ByteTrack callback alone between the
two probes.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadvision.config import RoadVisionConfig  # noqa: E402
from roadvision.pipeline import require_device  # noqa: E402


IMGSZ = 640
IOU = 0.7
DEFAULT_FRAMES = 100
WARMUP_FRAMES = 1
TIMINGS_NAME = "inference_timings.csv"
SUMMARY_NAME = "inference_summary.json"
REPORT_NAME = "inference_report.md"

TIMING_COLUMNS = (
    "timing_mode",
    "frame",
    "warmup",
    "tracked_objects",
    "track_call_ms",
    "ultralytics_preprocess_ms",
    "ultralytics_inference_ms",
    "ultralytics_postprocess_ms",
    "tracker_callback_ms",
    "tracker_callback_measured",
    "result_cpu_conversion_ms",
    "synced_preprocess_ms",
    "synced_inference_ms",
    "synced_postprocess_ms",
)
MEASURED_FIELDS = (
    "track_call_ms",
    "ultralytics_preprocess_ms",
    "ultralytics_inference_ms",
    "ultralytics_postprocess_ms",
    "tracker_callback_ms",
    "result_cpu_conversion_ms",
    "synced_preprocess_ms",
    "synced_inference_ms",
    "synced_postprocess_ms",
)
OUTER_SYNC = "outer_sync"
STAGE_SYNC = "stage_sync"


@dataclass
class BenchmarkSettings:
    """Inputs copied from the current RoadVision defaults."""

    model_path: str
    device: str
    confidence: float
    classes: tuple[int, ...]
    tracker_path: str
    video_path: str
    imgsz: int = IMGSZ
    iou: float = IOU
    max_frames: int = DEFAULT_FRAMES
    output_dir: Path = Path("runs/inference_diagnostics")


class TrackerProbe:
    """Timestamps placed immediately around the ByteTrack callback."""

    def __init__(self) -> None:
        self.start_ns: int | None = None
        self.end_ns: int | None = None

    def reset(self) -> None:
        self.start_ns = None
        self.end_ns = None

    def mark_start(self, predictor: object) -> None:
        self.start_ns = time.perf_counter_ns()

    def mark_end(self, predictor: object) -> None:
        self.end_ns = time.perf_counter_ns()

    def elapsed_ms(self) -> float | None:
        if self.start_ns is None or self.end_ns is None:
            return None
        if self.end_ns < self.start_ns:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure YOLO.track on decoded frames. "
            "This does not modify the RoadVision pipeline."
        )
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    settings = settings_from_defaults()
    if args.video is not None:
        settings.video_path = str(args.video)
    if args.output_dir is not None:
        settings.output_dir = args.output_dir
    if args.device is not None:
        settings.device = args.device
    settings.max_frames = args.max_frames
    if settings.max_frames < 2:
        print("max-frames must be at least 2.", file=sys.stderr)
        return 2

    require_device(settings.device)
    frames = decode_frames(Path(settings.video_path), settings.max_frames)
    if len(frames) < 2:
        print(
            f"Need at least 2 decoded frames from {settings.video_path}.",
            file=sys.stderr,
        )
        return 1

    outer_rows, outer_meta = run_pass(frames, settings, OUTER_SYNC)
    stage_error = None
    stage_rows: list[dict[str, object]] = []
    stage_meta: dict[str, object] | None = None
    try:
        stage_rows, stage_meta = run_pass(frames, settings, STAGE_SYNC)
    except Exception as exc:
        stage_error = f"{type(exc).__name__}: {exc}"

    summary = build_summary(
        settings,
        frames_decoded=len(frames),
        outer_rows=outer_rows,
        outer_meta=outer_meta,
        stage_rows=stage_rows,
        stage_meta=stage_meta,
        stage_error=stage_error,
    )
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_timings(output_dir / TIMINGS_NAME, [*outer_rows, *stage_rows])
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(summary)
    (output_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    print(f"Wrote {output_dir / REPORT_NAME}")
    return 0


def settings_from_defaults() -> BenchmarkSettings:
    """Use the application defaults without reading a profiled run."""

    config = RoadVisionConfig()
    return BenchmarkSettings(
        model_path=config.detector.model_path,
        device=config.detector.device,
        confidence=config.detector.confidence,
        classes=tuple(config.detector.classes),
        tracker_path=config.tracker.config_path,
        video_path=config.video.input_path,
    )


def decode_frames(path: Path, count: int) -> list[np.ndarray]:
    """Read BGR frames and release the video before any track() call."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def run_pass(
    frames: list[np.ndarray],
    settings: BenchmarkSettings,
    timing_mode: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build a new model and time ``YOLO.track`` on ``frames``."""

    from ultralytics import YOLO

    started = time.perf_counter_ns()
    model = YOLO(settings.model_path)
    construction_ms = (time.perf_counter_ns() - started) / 1_000_000
    predictor_cls = (
        build_stage_predictor() if timing_mode == STAGE_SYNC else None
    )
    kwargs = track_kwargs(settings, predictor_cls)
    probe: TrackerProbe | None = None
    probe_reason = "not installed until after the first track() call"
    rows: list[dict[str, object]] = []

    for number, frame in enumerate(frames, start=1):
        if probe is not None:
            probe.reset()
        bracket_clean = probe is not None and callback_bracket_is_exclusive(
            model.callbacks["on_predict_postprocess_end"],
            probe.mark_start,
            probe.mark_end,
            tracker_callback_func(),
        )
        synchronize(settings.device)
        call_started = time.perf_counter_ns()
        results = model.track(frame, **kwargs)
        synchronize(settings.device)
        call_ms = (time.perf_counter_ns() - call_started) / 1_000_000

        conversion_started = time.perf_counter_ns()
        tracked = convert_tracked_boxes(results[0])
        conversion_ms = (
            time.perf_counter_ns() - conversion_started
        ) / 1_000_000
        callback_ms = probe.elapsed_ms() if bracket_clean and probe else None
        if callback_ms is None:
            bracket_clean = False
        speed = dict(getattr(results[0], "speed", {}) or {})
        stage_ns = stage_times(model)
        rows.append(
            {
                "timing_mode": timing_mode,
                "frame": number,
                "warmup": number <= WARMUP_FRAMES,
                "tracked_objects": tracked,
                "track_call_ms": call_ms,
                "ultralytics_preprocess_ms": _optional_float(
                    speed.get("preprocess")
                ),
                "ultralytics_inference_ms": _optional_float(
                    speed.get("inference")
                ),
                "ultralytics_postprocess_ms": _optional_float(
                    speed.get("postprocess")
                ),
                "tracker_callback_ms": callback_ms,
                "tracker_callback_measured": bracket_clean,
                "result_cpu_conversion_ms": conversion_ms,
                "synced_preprocess_ms": stage_ns["preprocess"],
                "synced_inference_ms": stage_ns["inference"],
                "synced_postprocess_ms": stage_ns["postprocess"],
            }
        )
        if number == 1:
            probe, probe_reason = install_tracker_probe(model)

    return rows, {
        "model_construction_ms": construction_ms,
        "tracker_probe_reason": probe_reason,
        "tracker_callback_body": describe_tracker_callback(
            tracker_callback_func()
        ),
    }


def track_kwargs(
    settings: BenchmarkSettings,
    predictor_cls: type | None,
) -> dict[str, object]:
    """Match the pipeline call and make its implicit defaults explicit."""

    kwargs: dict[str, object] = {
        "persist": True,
        "tracker": settings.tracker_path,
        "conf": settings.confidence,
        "iou": settings.iou,
        "imgsz": settings.imgsz,
        "classes": list(settings.classes),
        "device": settings.device,
        "verbose": False,
    }
    if predictor_cls is not None:
        kwargs["predictor"] = predictor_cls
    return kwargs


def build_stage_predictor() -> type:
    """Predictor that synchronizes around the three Ultralytics stages.

    The normal ``python -m roadvision run`` path never uses this class.
    The extra synchronization changes scheduling, so its totals are a
    diagnostic measurement rather than the production stage time.
    """

    from ultralytics.models.yolo.detect import DetectionPredictor

    class StageSynchronizedPredictor(DetectionPredictor):
        def preprocess(self, im):
            return self._timed("preprocess", super().preprocess, im)

        def inference(self, im, *args, **kwargs):
            return self._timed(
                "inference",
                super().inference,
                im,
                *args,
                **kwargs,
            )

        def postprocess(self, preds, img, orig_imgs, **kwargs):
            return self._timed(
                "postprocess",
                super().postprocess,
                preds,
                img,
                orig_imgs,
                **kwargs,
            )

        def _timed(self, name: str, method, *args, **kwargs):
            synchronize(self.device)
            started = time.perf_counter_ns()
            result = method(*args, **kwargs)
            synchronize(self.device)
            setattr(
                self,
                f"_rv_{name}_ms",
                (time.perf_counter_ns() - started) / 1_000_000,
            )
            return result

    return StageSynchronizedPredictor


def stage_times(model: object) -> dict[str, float | None]:
    predictor = getattr(model, "predictor", None)
    return {
        "preprocess": _optional_float(
            getattr(predictor, "_rv_preprocess_ms", None)
        ),
        "inference": _optional_float(
            getattr(predictor, "_rv_inference_ms", None)
        ),
        "postprocess": _optional_float(
            getattr(predictor, "_rv_postprocess_ms", None)
        ),
    }


def convert_tracked_boxes(result: object) -> int:
    """Repeat the tensor copies made by ``YoloByteTrack.track``."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return 0
    coords = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.int().cpu().tolist()
    boxes.cls.int().cpu().tolist()
    boxes.conf.cpu().tolist()
    del coords
    return len(track_ids)


def install_tracker_probe(
    model: object,
    tracker_func: object | None = None,
) -> tuple[TrackerProbe | None, str]:
    """Insert probes only when they enclose the tracker callback alone."""

    predictor = getattr(model, "predictor", None)
    callbacks = model.callbacks["on_predict_postprocess_end"]
    if predictor is None or predictor.callbacks is not model.callbacks:
        return None, (
            "The predictor callback list is not the model callback list, "
            "so a probe on the model would not time the running callback."
        )
    if tracker_func is None:
        tracker_func = tracker_callback_func()
    indexes = [
        index
        for index, callback in enumerate(callbacks)
        if getattr(callback, "func", None) is tracker_func
    ]
    if len(indexes) != 1:
        return None, (
            "Expected one ByteTrack callback on "
            "on_predict_postprocess_end, "
            f"found {len(indexes)}."
        )
    probe = TrackerProbe()
    index = indexes[0]
    callbacks.insert(index, probe.mark_start)
    callbacks.insert(index + 2, probe.mark_end)
    if not callback_bracket_is_exclusive(
        callbacks,
        probe.mark_start,
        probe.mark_end,
        tracker_func,
    ):
        callbacks.remove(probe.mark_start)
        callbacks.remove(probe.mark_end)
        return None, (
            "The ByteTrack callback was not alone between the probes."
        )
    return probe, (
        "Probes enclose only the ByteTrack "
        "on_predict_postprocess_end callback."
    )


def callback_bracket_is_exclusive(
    callbacks: list,
    start: object,
    end: object,
    tracker_func: object,
) -> bool:
    """True when exactly the tracker callback sits between the probes."""

    try:
        start_index = callbacks.index(start)
        end_index = callbacks.index(end)
    except ValueError:
        return False
    between = callbacks[start_index + 1 : end_index]
    return (
        end_index == start_index + 2
        and len(between) == 1
        and getattr(between[0], "func", None) is tracker_func
    )


def tracker_callback_func():
    from ultralytics.trackers.track import (
        on_predict_postprocess_end as tracker_func,
    )

    return tracker_func


def describe_tracker_callback(func: object) -> dict[str, bool]:
    """Check the callback source for the two operations it should cover."""

    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = ""
    return {
        "copies_boxes_to_cpu": ".cpu()" in source and ".numpy()" in source,
        "calls_tracker_update": "tracker.update" in source,
    }


def synchronize(device: object) -> None:
    """Wait for MPS or CUDA. CPU is a no-op."""

    kind = device.type if hasattr(device, "type") else str(device).split(":")[0]
    if kind in {"cpu", "None", ""}:
        return
    import torch

    if kind == "mps":
        torch.mps.synchronize()
        return
    if kind == "cuda":
        torch.cuda.synchronize()


def build_summary(
    settings: BenchmarkSettings,
    *,
    frames_decoded: int,
    outer_rows: list[dict[str, object]],
    outer_meta: dict[str, object],
    stage_rows: list[dict[str, object]],
    stage_meta: dict[str, object] | None,
    stage_error: str | None,
) -> dict:
    outer = summarize_rows(outer_rows)
    stage = summarize_rows(stage_rows) if stage_rows else None
    outer_mean = _mean(outer, "track_call_ms")
    stage_mean = _mean(stage, "track_call_ms") if stage else None
    overhead = None
    if outer_mean is not None and stage_mean is not None:
        overhead = stage_mean - outer_mean
    return {
        "schema_version": "1",
        "environment": _environment(),
        "settings": {
            "model_path": settings.model_path,
            "device": settings.device,
            "confidence": settings.confidence,
            "iou": settings.iou,
            "imgsz": settings.imgsz,
            "classes": list(settings.classes),
            "tracker_path": settings.tracker_path,
            "video_path": settings.video_path,
            "frames_requested": settings.max_frames,
            "frames_decoded": frames_decoded,
            "input_color": "BGR uint8 from OpenCV",
        },
        "passes": {
            OUTER_SYNC: {**outer, **outer_meta},
            STAGE_SYNC: (
                {**stage, **(stage_meta or {})}
                if stage is not None
                else {"error": stage_error}
            ),
        },
        "synchronization_overhead": {
            "outer_track_call_mean_ms": outer_mean,
            "stage_sync_track_call_mean_ms": stage_mean,
            "added_mean_ms": overhead,
            "definition": (
                "Stage-synchronized mean minus the earlier "
                "single-synchronization mean. The passes use separate "
                "model instances, and the second starts after the first "
                "has paid the long first-frame warmup, so a negative "
                "value is not a synchronization saving."
            ),
        },
        "timing_definition": {
            "track_call_ms": (
                "perf_counter_ns around YOLO.track only. MPS or CUDA "
                "is synchronized immediately before the call and "
                "immediately after it returns. Decoding, model "
                "construction, and the later tensor conversion are "
                "outside this interval."
            ),
            "warmup": (
                "The first completed frame is stored and excluded from "
                "steady-state statistics. Its track() call includes "
                "Ultralytics model.warmup, which sits outside "
                "Results.speed."
            ),
            "ultralytics_speed": (
                "Results.speed preprocess, inference, and postprocess "
                "are Ultralytics Profile timings. On MPS they do not "
                "synchronize the device. Their sum is not the "
                "synchronized track() time, and the gap is not "
                "ByteTrack time."
            ),
            "tracker_callback_ms": (
                "Recorded only when mark_start, the ByteTrack "
                "on_predict_postprocess_end callback, and mark_end are "
                "adjacent. The interval includes that callback's "
                "box copy to CPU and tracker.update when the installed "
                "source contains those operations. It is empty on the "
                "first frame because the callback is registered during "
                "that first call."
            ),
            "result_cpu_conversion_ms": (
                "The .cpu() copies YoloByteTrack.track performs after "
                "YOLO.track returns. Those copies are outside "
                "track_call_ms in this diagnostic and inside the "
                "pipeline's inference_tracking_ms span."
            ),
            "stage_sync": (
                "A benchmark-only predictor synchronizes before and "
                "after preprocess, inference, and postprocess. Those "
                "fields are empty in outer_sync rows. The extra waits "
                "are included in that pass's track_call_ms."
            ),
        },
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict:
    """Aggregate steady-state rows. Warm-up rows stay in the CSV only."""

    measured = [row for row in rows if not row["warmup"]]
    warmup = [row for row in rows if row["warmup"]]
    fields = {}
    for name in MEASURED_FIELDS:
        values = [
            float(row[name])
            for row in measured
            if row[name] is not None
        ]
        fields[name] = distribution(values)
    callback_measured = sum(
        1 for row in measured if row["tracker_callback_measured"]
    )
    return {
        "recorded_frames": len(rows),
        "measured_frames": len(measured),
        "warmup_frames": len(warmup),
        "warmup_track_call_ms": (
            float(warmup[0]["track_call_ms"]) if warmup else None
        ),
        "warmup_ultralytics_inference_ms": (
            _optional_float(warmup[0]["ultralytics_inference_ms"])
            if warmup
            else None
        ),
        "tracker_callback_measured_frames": callback_measured,
        "fields": fields,
    }


def distribution(values: list[float]) -> dict[str, float | int | None]:
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


def render_report(summary: dict) -> str:
    outer = summary["passes"][OUTER_SYNC]
    stage = summary["passes"][STAGE_SYNC]
    settings = summary["settings"]
    environment = summary["environment"]
    lines = [
        "# Inference diagnostics",
        "",
        "Decoded BGR frames were timed with a new YOLO model and a new "
        "ByteTrack state. The RoadVision pipeline was not used.",
        "",
        "## Settings",
        "",
        (
            f"- {environment['python']} on {environment['operating_system']} "
            f"({environment['cpu_architecture']})"
        ),
        (
            f"- Torch {environment['torch']}, "
            f"Ultralytics {environment['ultralytics']}"
        ),
        (
            f"- Model `{settings['model_path']}` on `{settings['device']}`, "
            f"imgsz {settings['imgsz']}, confidence {settings['confidence']}, "
            f"IoU {settings['iou']}"
        ),
        f"- Classes: {settings['classes']}",
        f"- Tracker: `{settings['tracker_path']}`",
        (
            f"- Video: `{settings['video_path']}`, "
            f"{settings['frames_decoded']} decoded {settings['input_color']} frames"
        ),
        "",
        "## Synchronized wall-clock timing",
        "",
        summary["timing_definition"]["track_call_ms"],
        "",
        _pass_table(outer, "track_call_ms", "YOLO.track"),
        "",
        (
            f"Model construction for this pass: "
            f"{outer['model_construction_ms']:.3f} ms. "
            "It is outside the frame rows."
        ),
        "",
        "## Ultralytics-reported internal timing",
        "",
        summary["timing_definition"]["ultralytics_speed"],
        "",
        _speed_table(outer),
        "",
        "## First-frame initialization",
        "",
        summary["timing_definition"]["warmup"],
        "",
        _warmup_sentence(outer),
        "",
        "## Tracking callback and tensor conversion",
        "",
        summary["timing_definition"]["tracker_callback_ms"],
        "",
        _callback_section(outer),
        "",
        summary["timing_definition"]["result_cpu_conversion_ms"],
        "",
        _pass_table(
            outer,
            "result_cpu_conversion_ms",
            "Result conversion after track()",
        ),
        "",
        "## Stage-synchronized diagnostic",
        "",
        summary["timing_definition"]["stage_sync"],
        "",
        _stage_section(stage, summary["synchronization_overhead"]),
        "",
        "## Timing ambiguity",
        "",
        _ambiguity_section(summary),
        "",
        "## Reading for a backend experiment",
        "",
        _recommendation(summary),
        "",
    ]
    return "\n".join(lines)


def write_timings(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS)
        writer.writeheader()
        for row in rows:
            stored = {}
            for column in TIMING_COLUMNS:
                value = row[column]
                if isinstance(value, float):
                    stored[column] = f"{value:.6f}"
                elif isinstance(value, bool):
                    stored[column] = "true" if value else "false"
                elif value is None:
                    stored[column] = ""
                else:
                    stored[column] = value
            writer.writerow(stored)


def _pass_table(pass_summary: dict, field: str, label: str) -> str:
    stats = pass_summary["fields"][field]
    if stats["count"] == 0:
        return f"{label}: no steady-state measurements."
    return (
        f"{label}, {stats['count']} steady-state frames: "
        f"mean {stats['mean_ms']:.3f} ms, "
        f"median {stats['median_ms']:.3f} ms, "
        f"p95 {stats['p95_ms']:.3f} ms, "
        f"p99 {stats['p99_ms']:.3f} ms, "
        f"min {stats['min_ms']:.3f} ms, "
        f"max {stats['max_ms']:.3f} ms."
    )


def _speed_table(pass_summary: dict) -> str:
    lines = [
        "| Ultralytics field | Count | Mean ms | Median ms | p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "ultralytics_preprocess_ms",
        "ultralytics_inference_ms",
        "ultralytics_postprocess_ms",
    ):
        stats = pass_summary["fields"][name]
        lines.append(
            f"| `{name}` | {stats['count']} | {_num(stats['mean_ms'])} | "
            f"{_num(stats['median_ms'])} | {_num(stats['p95_ms'])} |"
        )
    return "\n".join(lines)


def _callback_section(pass_summary: dict) -> str:
    body = pass_summary["tracker_callback_body"]
    reason = pass_summary["tracker_probe_reason"]
    stats = pass_summary["fields"]["tracker_callback_ms"]
    confirmed = (
        body["copies_boxes_to_cpu"] and body["calls_tracker_update"]
    )
    lines = [
        f"Probe status: {reason}",
        (
            "Installed callback source contains the box copy to CPU: "
            f"{body['copies_boxes_to_cpu']}. It contains `tracker.update`: "
            f"{body['calls_tracker_update']}."
        ),
    ]
    if stats["count"] and confirmed:
        lines.append(
            "Steady-state `tracker_callback_ms` covers that callback only: "
            f"mean {stats['mean_ms']:.3f} ms, "
            f"median {stats['median_ms']:.3f} ms, "
            f"p95 {stats['p95_ms']:.3f} ms, "
            f"on {pass_summary['tracker_callback_measured_frames']} frames."
        )
    elif stats["count"] == 0:
        lines.append(
            "`tracker_callback_ms` was not recorded. "
            "No remainder is substituted for it."
        )
    else:
        lines.append(
            "The interval was recorded, but the callback source did not "
            "confirm both operations, so it is not described as the "
            "complete tracker body."
        )
    return "\n\n".join(lines)


def _stage_section(stage: dict, overhead: dict) -> str:
    if "fields" not in stage:
        return f"Stage-synchronized pass did not complete: {stage.get('error')}"
    lines = [
        _pass_table(stage, "synced_preprocess_ms", "Synchronized preprocess"),
        _pass_table(stage, "synced_inference_ms", "Synchronized forward pass"),
        _pass_table(stage, "synced_postprocess_ms", "Synchronized postprocess"),
        _pass_table(
            stage,
            "track_call_ms",
            "YOLO.track with the extra synchronization inside it",
        ),
    ]
    added = overhead["added_mean_ms"]
    if added is not None:
        relation = "higher" if added > 0 else "lower"
        lines.append(
            "The stage-synchronized steady-state mean is "
            f"{abs(added):.3f} ms {relation} than the earlier "
            "single-synchronization mean "
            f"({overhead['stage_sync_track_call_mean_ms']:.3f} ms versus "
            f"{overhead['outer_track_call_mean_ms']:.3f} ms). "
            "These are separate model instances. The second pass starts "
            "after the first has already paid the long first-frame "
            "warmup, so this gap is not an isolated synchronization cost."
        )
    return "\n\n".join(lines)


def _ambiguity_section(summary: dict) -> str:
    outer = summary["passes"][OUTER_SYNC]
    speed_sum = _sum_means(
        outer,
        (
            "ultralytics_preprocess_ms",
            "ultralytics_inference_ms",
            "ultralytics_postprocess_ms",
        ),
    )
    track_mean = _mean(outer, "track_call_ms")
    lines = [
        "The three Ultralytics fields are not subtracted from "
        "`track_call_ms`. The arithmetic gap is not a ByteTrack duration.",
    ]
    if speed_sum is not None and track_mean is not None:
        lines.append(
            f"For reference only, their steady-state means add to {speed_sum:.3f} ms "
            f"beside a synchronized `YOLO.track` mean of {track_mean:.3f} ms."
        )
    stage = summary["passes"][STAGE_SYNC]
    if "fields" in stage:
        unsynchronized_inference = _mean(outer, "ultralytics_inference_ms")
        unsynchronized_postprocess = _mean(outer, "ultralytics_postprocess_ms")
        synchronized_inference = _mean(stage, "synced_inference_ms")
        synchronized_postprocess = _mean(stage, "synced_postprocess_ms")
        if None not in (
            unsynchronized_inference,
            unsynchronized_postprocess,
            synchronized_inference,
            synchronized_postprocess,
        ):
            lines.append(
                "Ultralytics' unsynchronized means rank postprocess at "
                f"{unsynchronized_postprocess:.3f} ms and inference at "
                f"{unsynchronized_inference:.3f} ms. Synchronizing each "
                "method reverses that ranking: forward "
                f"{synchronized_inference:.3f} ms and postprocess "
                f"{synchronized_postprocess:.3f} ms. The unsynchronized "
                "postprocess interval is where unfinished MPS work "
                "becomes visible."
            )
        accounted = _sum_means(
            stage,
            (
                "synced_preprocess_ms",
                "synced_inference_ms",
                "synced_postprocess_ms",
                "tracker_callback_ms",
            ),
        )
        stage_track = _mean(stage, "track_call_ms")
        if accounted is not None and stage_track is not None:
            lines.append(
                "Inside the stage-synchronized pass, preprocess, the "
                "forward pass, postprocess, and the tracker callback "
                f"average {accounted:.3f} ms of the {stage_track:.3f} ms "
                "call. The rest is other Python work inside `track()`, "
                "including source setup. It is not a second tracker measurement."
            )
    return "\n\n".join(lines)


def _warmup_sentence(pass_summary: dict) -> str:
    text = (
        "First-frame synchronized `YOLO.track`: "
        f"{pass_summary['warmup_track_call_ms']:.3f} ms."
    )
    reported = pass_summary.get("warmup_ultralytics_inference_ms")
    if reported is not None:
        text += (
            " Ultralytics reports "
            f"{reported:.3f} ms of inference inside that same call, "
            "so the long first call is not the `Results.speed` inference field."
        )
    return text


def _recommendation(summary: dict) -> str:
    stage = summary["passes"][STAGE_SYNC]
    outer = summary["passes"][OUTER_SYNC]
    if "fields" not in stage:
        return (
            "The forward pass could not be separated with synchronized "
            "stage timers. ONNX Runtime remains the first backend to "
            "compare because it can keep this Python NMS and ByteTrack "
            "path and can run on Linux. Core ML remains Mac-only."
        )
    inference = _mean(stage, "synced_inference_ms")
    named = [
        (
            "synchronized preprocess and postprocess",
            _sum_means(
                stage,
                ("synced_preprocess_ms", "synced_postprocess_ms"),
            ),
        ),
        ("measured tracker callback", _mean(outer, "tracker_callback_ms")),
        (
            "wrapper conversion",
            _mean(outer, "result_cpu_conversion_ms"),
        ),
    ]
    present = [(label, value) for label, value in named if value is not None]
    other_total = sum(value for _, value in present) if present else None
    other_label = _joined_labels([label for label, _ in present])
    if (
        inference is not None
        and other_total is not None
        and inference > other_total
    ):
        ranking = (
            f"The synchronized forward pass averages {inference:.3f} ms, "
            f"which is larger than {other_label} combined "
            f"({other_total:.3f} ms)."
        )
    elif inference is not None:
        ranking = (
            f"The synchronized forward pass averages {inference:.3f} ms. "
            f"It is not larger than {other_label or 'the other measured pieces'} "
            "combined, so a backend export is not the only relevant cost."
        )
    else:
        ranking = "The synchronized forward pass has no steady-state samples."
    return (
        f"{ranking} ONNX Runtime remains the first backend experiment: "
        "export with the same weights, imgsz, and external NMS, then "
        "compare boxes before comparing tracks. Core ML stays a later, "
        "Mac-only measurement."
    )


def _sum_means(pass_summary: dict | None, names: tuple[str, ...]) -> float | None:
    if pass_summary is None:
        return None
    total = 0.0
    for name in names:
        value = _mean(pass_summary, name)
        if value is None:
            return None
        total += value
    return total


def _mean(pass_summary: dict | None, field: str) -> float | None:
    if pass_summary is None or "fields" not in pass_summary:
        return None
    value = pass_summary["fields"][field]["mean_ms"]
    if value is None:
        return None
    return float(value)


def _environment() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "cpu_architecture": platform.machine(),
        "torch": _package_version("torch"),
        "ultralytics": _package_version("ultralytics"),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _round(value: float) -> float:
    return round(value, 6)


def _joined_labels(labels: list[str]) -> str:
    if not labels:
        return "the other measured pieces"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
