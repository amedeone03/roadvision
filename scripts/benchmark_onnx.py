"""Compare PyTorch/MPS and ONNX Runtime CPU on the same decoded frames.

This is an experiment. It does not change RoadVision's pipeline,
thresholds, or default model. Both timed calls are complete
Ultralytics ``predict`` or ``track`` wall-clock measurements.
The PyTorch side synchronizes MPS around the call. The ONNX side
uses host time. Neither number is a GPU kernel duration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import shutil
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadvision.config import RoadVisionConfig  # noqa: E402


IMGSZ = 640
IOU = 0.7
MAX_DET = 300
CONFIDENCE = 0.25
MATCH_IOU = 0.5
NEAR_CONFIDENCE = 0.30
NOTICEABLE_CONFIDENCE_GAP = 0.02
DEFAULT_FRAMES = 100
WARMUP_FRAMES = 1
PYTORCH = "pytorch_mps"
ONNX = "onnx_cpu"

EXPORT_SETTINGS = {
    "format": "onnx",
    "imgsz": IMGSZ,
    "dynamic": False,
    "nms": False,
    "simplify": False,
    "quantize": None,
}

LATENCY_COLUMNS = (
    "engine",
    "call",
    "frame",
    "warmup",
    "elapsed_ms",
    "ultralytics_preprocess_ms",
    "ultralytics_inference_ms",
    "ultralytics_postprocess_ms",
    "detections",
)
MATCH_COLUMNS = (
    "call",
    "frame",
    "reference_detections",
    "candidate_detections",
    "matched",
    "unmatched_reference",
    "unmatched_candidate",
    "same_track_id",
    "track_id_comparisons",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export YOLO11n to ONNX and compare it with PyTorch/MPS. "
            "The RoadVision pipeline is not modified."
        )
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/onnx_benchmark"))
    parser.add_argument("--max-frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args(argv)

    runtime = probe_onnx_runtime()
    if not runtime["available"]:
        print(
            "ONNX Runtime is not installed in this interpreter. "
            "Use the isolated ONNX environment. The main RoadVision "
            "environment is intentionally unchanged.",
            file=sys.stderr,
        )
        return 2
    if args.max_frames < 2:
        print("max-frames must be at least 2.", file=sys.stderr)
        return 2

    config = RoadVisionConfig()
    weights = args.weights or Path(config.detector.model_path)
    video = args.video or Path(config.video.input_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    export_settings = dict(EXPORT_SETTINGS)
    export_settings["opset"] = args.opset
    problems = validate_export_settings(export_settings)
    if problems:
        print(f"Refusing export settings: {', '.join(problems)}", file=sys.stderr)
        return 2

    frames = decode_frames(video, args.max_frames)
    if len(frames) < 2:
        print(f"Need at least 2 frames from {video}.", file=sys.stderr)
        return 1

    artifact = export_onnx(
        weights,
        output_dir / "yolo11n.onnx",
        export_settings,
        force=args.force_export,
    )
    inspection = inspect_onnx_artifact(artifact)
    if not inspection["external_nms"]:
        print(
            f"ONNX output {inspection['outputs']} looks like embedded NMS.",
            file=sys.stderr,
        )
        return 1

    common = prediction_kwargs(config)
    pytorch_predict, pytorch_load = time_call_sequence(
        weights,
        frames,
        {**common, "device": "mps"},
        call="predict",
        engine=PYTORCH,
    )
    onnx_predict, onnx_load = time_call_sequence(
        artifact,
        frames,
        {**common, "device": "cpu"},
        call="predict",
        engine=ONNX,
    )
    detection = compare_sequences(
        [row["boxes"] for row in pytorch_predict],
        [row["boxes"] for row in onnx_predict],
    )
    pytorch_track, _ = time_call_sequence(
        weights,
        frames,
        {
            **common,
            "device": "mps",
            "persist": True,
            "tracker": config.tracker.config_path,
        },
        call="track",
        engine=PYTORCH,
    )
    onnx_track, _ = time_call_sequence(
        artifact,
        frames,
        {
            **common,
            "device": "cpu",
            "persist": True,
            "tracker": config.tracker.config_path,
        },
        call="track",
        engine=ONNX,
    )
    tracking = compare_sequences(
        [row["boxes"] for row in pytorch_track],
        [row["boxes"] for row in onnx_track],
    )
    summary = build_summary(
        config=config,
        weights=weights,
        video=video,
        frames=len(frames),
        export_settings=export_settings,
        artifact=artifact,
        inspection=inspection,
        runtime=runtime,
        pytorch_load_ms=pytorch_load,
        onnx_load_ms=onnx_load,
        pytorch_predict=pytorch_predict,
        onnx_predict=onnx_predict,
        pytorch_track=pytorch_track,
        onnx_track=onnx_track,
        detection=detection,
        tracking=tracking,
    )
    write_outputs(output_dir, summary, pytorch_predict, onnx_predict, pytorch_track, onnx_track)
    print(f"Wrote {output_dir / 'onnx_report.md'}")
    return 0


def prediction_kwargs(config: RoadVisionConfig) -> dict[str, object]:
    """Thresholds shared by both engines. Tracking is added by the caller."""

    return {
        "conf": config.detector.confidence,
        "iou": IOU,
        "imgsz": IMGSZ,
        "classes": list(config.detector.classes),
        "max_det": MAX_DET,
        "verbose": False,
    }


def validate_export_settings(settings: dict) -> list[str]:
    """Return the names of export choices this experiment does not allow."""

    problems = []
    if settings.get("format") != "onnx":
        problems.append("format")
    if settings.get("imgsz") != IMGSZ:
        problems.append("imgsz")
    if settings.get("dynamic") is not False:
        problems.append("dynamic")
    if settings.get("nms") is not False:
        problems.append("nms")
    if settings.get("simplify") is not False:
        problems.append("simplify")
    if settings.get("quantize") not in (None, 32, "32", "fp32"):
        problems.append("quantize")
    opset = settings.get("opset")
    if opset is not None and (not isinstance(opset, int) or isinstance(opset, bool) or opset < 9):
        problems.append("opset")
    return problems


def probe_onnx_runtime() -> dict[str, object]:
    """Report ONNX Runtime without raising when it is absent."""

    if importlib.util.find_spec("onnxruntime") is None:
        return {"available": False, "version": None, "providers": []}
    import onnxruntime

    return {
        "available": True,
        "version": onnxruntime.__version__,
        "providers": list(onnxruntime.get_available_providers()),
    }


def decode_frames(path: Path, count: int) -> list[np.ndarray]:
    """Read BGR frames and close the video before inference starts."""

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


def export_onnx(
    weights: Path,
    destination: Path,
    settings: dict,
    *,
    force: bool = False,
) -> Path:
    """Export FP32 ONNX into the ignored benchmark directory."""

    if destination.is_file() and not force:
        return destination

    from ultralytics import YOLO

    export_kwargs = {
        "format": "onnx",
        "imgsz": settings["imgsz"],
        "dynamic": False,
        "nms": False,
        "simplify": False,
        "device": "cpu",
    }
    if settings.get("opset") is not None:
        export_kwargs["opset"] = settings["opset"]
    produced = Path(YOLO(str(weights)).export(**export_kwargs))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != destination.resolve():
        shutil.move(str(produced), str(destination))
    return destination


def inspect_onnx_artifact(path: Path) -> dict[str, object]:
    """Load the graph with the CPU provider and describe its interface."""

    import onnx
    import onnxruntime

    model = onnx.load(str(path))
    opsets = [
        {"domain": item.domain or "ai.onnx", "version": int(item.version)}
        for item in model.opset_import
    ]
    session = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    inputs = [_tensor_info(item) for item in session.get_inputs()]
    outputs = [_tensor_info(item) for item in session.get_outputs()]
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "opsets": opsets,
        "inputs": inputs,
        "outputs": outputs,
        "session_providers": list(session.get_providers()),
        "external_nms": _nms_is_external(outputs),
    }


def time_call_sequence(
    model_path: Path,
    frames: list[np.ndarray],
    kwargs: dict[str, object],
    *,
    call: str,
    engine: str,
) -> tuple[list[dict[str, object]], float]:
    """Time one fresh model. Frame 1 remains the warm-up row."""

    from ultralytics import YOLO

    started = time.perf_counter_ns()
    model = YOLO(str(model_path))
    load_ms = (time.perf_counter_ns() - started) / 1_000_000
    device = str(kwargs["device"])
    method = model.predict if call == "predict" else model.track
    rows = []
    for number, frame in enumerate(frames, start=1):
        synchronize(device)
        call_started = time.perf_counter_ns()
        results = method(frame, **kwargs)
        synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - call_started) / 1_000_000
        result = results[0]
        speed = dict(getattr(result, "speed", {}) or {})
        boxes = boxes_from_result(result, include_track_id=call == "track")
        rows.append(
            {
                "engine": engine,
                "call": call,
                "frame": number,
                "warmup": number <= WARMUP_FRAMES,
                "elapsed_ms": elapsed_ms,
                "ultralytics_preprocess_ms": _optional_float(speed.get("preprocess")),
                "ultralytics_inference_ms": _optional_float(speed.get("inference")),
                "ultralytics_postprocess_ms": _optional_float(speed.get("postprocess")),
                "detections": len(boxes),
                "boxes": boxes,
            }
        )
    return rows, load_ms


def boxes_from_result(result: object, *, include_track_id: bool) -> list[dict]:
    """Read boxes after Ultralytics NMS. This work is outside the timer."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _as_numpy(boxes.xyxy)
    confidence = _as_numpy(boxes.conf)
    classes = _as_numpy(boxes.cls)
    identities = None
    if include_track_id and boxes.id is not None:
        identities = _as_numpy(boxes.id)
    parsed = []
    for index, row in enumerate(xyxy):
        item = {
            "cls": int(classes[index]),
            "conf": float(confidence[index]),
            "xyxy": tuple(float(value) for value in row),
        }
        if identities is not None:
            item["track_id"] = int(identities[index])
        parsed.append(item)
    return parsed


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def compare_sequences(
    reference_frames: list[list[dict]],
    candidate_frames: list[list[dict]],
) -> dict:
    """Match each frame independently. PyTorch is the reference."""

    per_frame = [
        match_boxes(reference, candidate)
        for reference, candidate in zip(reference_frames, candidate_frames)
    ]
    reference_ids = [
        [int(box["track_id"]) for box in frame if "track_id" in box]
        for frame in reference_frames
    ]
    candidate_ids = [
        [int(box["track_id"]) for box in frame if "track_id" in box]
        for frame in candidate_frames
    ]
    has_ids = any(reference_ids) or any(candidate_ids)
    return {
        "detection": summarize_matches(per_frame),
        "frames": [
            {
                "frame": index,
                "reference_detections": len(reference_frames[index - 1]),
                "candidate_detections": len(candidate_frames[index - 1]),
                "matched": frame["matched"],
                "unmatched_reference": frame["unmatched_reference"],
                "unmatched_candidate": frame["unmatched_candidate"],
                "same_track_id": frame["same_track_id"],
                "track_id_comparisons": frame["track_id_comparisons"],
            }
            for index, frame in enumerate(per_frame, start=1)
        ],
        "identity": compare_track_identity(reference_ids, candidate_ids) if has_ids else None,
    }


def match_boxes(
    reference: list[dict],
    candidate: list[dict],
    iou_threshold: float = MATCH_IOU,
) -> dict:
    """Greedy one-to-one matching inside each class, highest IoU first."""

    pairs = []
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    proposals = []
    for ref_index, ref_box in enumerate(reference):
        for cand_index, cand_box in enumerate(candidate):
            if ref_box["cls"] != cand_box["cls"]:
                continue
            overlap = box_iou(ref_box["xyxy"], cand_box["xyxy"])
            if overlap >= iou_threshold:
                proposals.append((overlap, ref_index, cand_index))
    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    for overlap, ref_index, cand_index in proposals:
        if ref_index in used_reference or cand_index in used_candidate:
            continue
        used_reference.add(ref_index)
        used_candidate.add(cand_index)
        ref_box = reference[ref_index]
        cand_box = candidate[cand_index]
        pair = {
            "iou": overlap,
            "center_px": center_distance(ref_box["xyxy"], cand_box["xyxy"]),
            "confidence_abs_diff": abs(ref_box["conf"] - cand_box["conf"]),
            "reference_conf": ref_box["conf"],
            "candidate_conf": cand_box["conf"],
            "cls": ref_box["cls"],
        }
        if "track_id" in ref_box and "track_id" in cand_box:
            pair["reference_track_id"] = ref_box["track_id"]
            pair["candidate_track_id"] = cand_box["track_id"]
            pair["same_track_id"] = ref_box["track_id"] == cand_box["track_id"]
        pairs.append(pair)
    unmatched_reference = [
        reference[index] for index in range(len(reference)) if index not in used_reference
    ]
    unmatched_candidate = [
        candidate[index] for index in range(len(candidate)) if index not in used_candidate
    ]
    id_pairs = [pair for pair in pairs if "same_track_id" in pair]
    return {
        "pairs": pairs,
        "matched": len(pairs),
        "unmatched_reference": len(unmatched_reference),
        "unmatched_candidate": len(unmatched_candidate),
        "unmatched_reference_boxes": unmatched_reference,
        "unmatched_candidate_boxes": unmatched_candidate,
        "same_track_id": sum(1 for pair in id_pairs if pair["same_track_id"]),
        "track_id_comparisons": len(id_pairs),
    }


def summarize_matches(frames: list[dict]) -> dict:
    ious = [pair["iou"] for frame in frames for pair in frame["pairs"]]
    centers = [pair["center_px"] for frame in frames for pair in frame["pairs"]]
    confidence_gaps = [
        pair["confidence_abs_diff"] for frame in frames for pair in frame["pairs"]
    ]
    near_unmatched = 0
    for frame in frames:
        for box in frame["unmatched_reference_boxes"] + frame["unmatched_candidate_boxes"]:
            if box["conf"] <= NEAR_CONFIDENCE:
                near_unmatched += 1
    near_gaps = sum(
        1
        for frame in frames
        for pair in frame["pairs"]
        if min(pair["reference_conf"], pair["candidate_conf"]) <= NEAR_CONFIDENCE
        and pair["confidence_abs_diff"] >= NOTICEABLE_CONFIDENCE_GAP
    )
    return {
        "reference_boxes": sum(
            frame["matched"] + frame["unmatched_reference"] for frame in frames
        ),
        "candidate_boxes": sum(
            frame["matched"] + frame["unmatched_candidate"] for frame in frames
        ),
        "matched": sum(frame["matched"] for frame in frames),
        "unmatched_reference": sum(frame["unmatched_reference"] for frame in frames),
        "unmatched_candidate": sum(frame["unmatched_candidate"] for frame in frames),
        "median_iou": _median(ious),
        "minimum_iou": min(ious) if ious else None,
        "median_center_px": _median(centers),
        "maximum_center_px": max(centers) if centers else None,
        "median_confidence_abs_diff": _median(confidence_gaps),
        "maximum_confidence_abs_diff": max(confidence_gaps) if confidence_gaps else None,
        "near_threshold_unmatched": near_unmatched,
        "near_threshold_confidence_gaps": near_gaps,
        "near_confidence_cutoff": NEAR_CONFIDENCE,
        "match_iou": MATCH_IOU,
        "same_track_id": sum(frame["same_track_id"] for frame in frames),
        "track_id_comparisons": sum(frame["track_id_comparisons"] for frame in frames),
    }


def compare_track_identity(
    reference_ids: list[list[int]],
    candidate_ids: list[list[int]],
) -> dict:
    """Describe identity continuity without requiring the IDs to be equal."""

    frames_with_different_ids = 0
    only_reference = 0
    only_candidate = 0
    for reference, candidate in zip(reference_ids, candidate_ids):
        ref_set = set(reference)
        cand_set = set(candidate)
        if ref_set != cand_set:
            frames_with_different_ids += 1
        only_reference += len(ref_set - cand_set)
        only_candidate += len(cand_set - ref_set)
    return {
        "reference": fragment_summary(reference_ids),
        "candidate": fragment_summary(candidate_ids),
        "frames_with_different_id_sets": frames_with_different_ids,
        "id_occurrences_only_in_reference": only_reference,
        "id_occurrences_only_in_candidate": only_candidate,
    }


def fragment_summary(frame_ids: list[list[int]]) -> dict:
    """Count a track again when its ID returns after at least one absence."""

    previous: set[int] = set()
    fragments: dict[int, int] = {}
    for identities in frame_ids:
        current = set(identities)
        for track_id in current - previous:
            fragments[track_id] = fragments.get(track_id, 0) + 1
        previous = current
    return {
        "unique_ids": len(fragments),
        "fragments": int(sum(fragments.values())),
        "ids_with_multiple_fragments": sum(
            1 for count in fragments.values() if count > 1
        ),
    }


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def center_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return math.hypot(
        ((left[0] + left[2]) / 2.0) - ((right[0] + right[2]) / 2.0),
        ((left[1] + left[3]) / 2.0) - ((right[1] + right[3]) / 2.0),
    )


def synchronize(device: object) -> None:
    kind = device.type if hasattr(device, "type") else str(device).split(":")[0]
    if kind in {"cpu", "None", ""}:
        return
    import torch

    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()


def build_summary(
    *,
    config: RoadVisionConfig,
    weights: Path,
    video: Path,
    frames: int,
    export_settings: dict,
    artifact: Path,
    inspection: dict,
    runtime: dict,
    pytorch_load_ms: float,
    onnx_load_ms: float,
    pytorch_predict: list[dict],
    onnx_predict: list[dict],
    pytorch_track: list[dict],
    onnx_track: list[dict],
    detection: dict,
    tracking: dict,
) -> dict:
    return {
        "schema_version": "1",
        "environment": {
            "python": platform.python_version(),
            "operating_system": platform.platform(),
            "cpu_architecture": platform.machine(),
            "torch": _package_version("torch"),
            "ultralytics": _package_version("ultralytics"),
            "numpy": _package_version("numpy"),
            "onnx": _package_version("onnx"),
            "onnxruntime": runtime["version"],
            "available_providers": runtime["providers"],
        },
        "settings": {
            "weights": str(weights),
            "video": str(video),
            "frames_decoded": frames,
            "imgsz": IMGSZ,
            "confidence": config.detector.confidence,
            "iou": IOU,
            "max_det": MAX_DET,
            "classes": list(config.detector.classes),
            "tracker": config.tracker.config_path,
            "pytorch_device": "mps",
            "onnx_device": "cpu",
            "onnx_provider": "CPUExecutionProvider",
        },
        "export": {
            "requested": {
                key: export_settings[key]
                for key in ("format", "imgsz", "dynamic", "nms", "simplify", "quantize", "opset")
            },
            "artifact": str(artifact),
            "inspection": inspection,
        },
        "latency": {
            "pytorch_mps": {
                "load_ms": round(pytorch_load_ms, 6),
                "predict": _latency_block(pytorch_predict),
                "track": _latency_block(pytorch_track),
            },
            "onnx_cpu": {
                "load_ms": round(onnx_load_ms, 6),
                "predict": _latency_block(onnx_predict),
                "track": _latency_block(onnx_track),
            },
        },
        "detection": detection["detection"],
        "tracking": {
            "boxes": tracking["detection"],
            "identity": tracking["identity"],
        },
        "timing_definition": {
            "elapsed_ms": (
                "Wall clock around one Ultralytics predict() or track() "
                "call. Decoding and box extraction are outside it. "
                "PyTorch/MPS synchronizes immediately before and after "
                "the call. ONNX Runtime CPU does not. The value includes "
                "Ultralytics preprocessing, the engine forward pass, and "
                "Python NMS. It is not a GPU kernel time."
            ),
            "warmup": (
                "The first frame of each fresh model is stored and "
                "excluded from steady-state mean, median, p95, p99, "
                "and throughput."
            ),
            "ultralytics_speed": (
                "Results.speed is recorded beside the wall clock. On "
                "MPS it is not an independently synchronized split."
            ),
            "matching": (
                "Boxes are matched within the same class by greedy "
                "one-to-one assignment, highest IoU first, at IoU >= 0.5. "
                "PyTorch/MPS is the reference. Equal track IDs are not "
                "required."
            ),
        },
    }


def write_outputs(
    output_dir: Path,
    summary: dict,
    pytorch_predict: list[dict],
    onnx_predict: list[dict],
    pytorch_track: list[dict],
    onnx_track: list[dict],
) -> None:
    _write_latency(
        output_dir / "latency.csv",
        [*pytorch_predict, *onnx_predict, *pytorch_track, *onnx_track],
    )
    _write_matches(
        output_dir / "detection_matches.csv",
        [
            *summary_frames_from_runs(pytorch_predict, onnx_predict, "predict"),
            *summary_frames_from_runs(pytorch_track, onnx_track, "track"),
        ],
    )
    (output_dir / "onnx_summary.json").write_text(
        json.dumps(_public_summary(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "onnx_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )


def summarize_latency(rows: list[dict]) -> dict:
    measured = [row for row in rows if not row["warmup"]]
    warmup = [row for row in rows if row["warmup"]]
    elapsed = distribution([float(row["elapsed_ms"]) for row in measured])
    mean = elapsed["mean_ms"]
    return {
        "measured_frames": len(measured),
        "warmup_ms": float(warmup[0]["elapsed_ms"]) if warmup else None,
        "elapsed": elapsed,
        "steady_frames_per_second": None if not mean else round(1000.0 / mean, 6),
        "ultralytics_preprocess_ms": distribution(
            [float(row["ultralytics_preprocess_ms"]) for row in measured if row["ultralytics_preprocess_ms"] is not None]
        ),
        "ultralytics_inference_ms": distribution(
            [float(row["ultralytics_inference_ms"]) for row in measured if row["ultralytics_inference_ms"] is not None]
        ),
        "ultralytics_postprocess_ms": distribution(
            [float(row["ultralytics_postprocess_ms"]) for row in measured if row["ultralytics_postprocess_ms"] is not None]
        ),
    }


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
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
        "mean_ms": round(float(array.mean()), 6),
        "median_ms": round(float(np.percentile(array, 50, method="linear")), 6),
        "p95_ms": round(float(np.percentile(array, 95, method="linear")), 6),
        "p99_ms": round(float(np.percentile(array, 99, method="linear")), 6),
        "min_ms": round(float(array.min()), 6),
        "max_ms": round(float(array.max()), 6),
    }


def render_report(summary: dict) -> str:
    env = summary["environment"]
    export = summary["export"]
    inspection = export["inspection"]
    pytorch = summary["latency"]["pytorch_mps"]
    onnx = summary["latency"]["onnx_cpu"]
    detection = summary["detection"]
    tracking = summary["tracking"]
    lines = [
        "# ONNX Runtime benchmark",
        "",
        "Experimental comparison only. The RoadVision pipeline and its "
        "thresholds were not changed. PyTorch and ONNX wall-clock times "
        "both cover an Ultralytics call, including preprocessing and "
        "Python NMS. They are not GPU-kernel measurements, and the "
        "earlier diagnostic stage times are not added to these figures.",
        "",
        "## Environment",
        "",
        f"- Python {env['python']} on {env['operating_system']} ({env['cpu_architecture']})",
        f"- Torch {env['torch']}, Ultralytics {env['ultralytics']}, NumPy {env['numpy']}",
        f"- ONNX {env['onnx']}, ONNX Runtime {env['onnxruntime']}",
        f"- Available providers: {', '.join(env['available_providers']) or 'none'}",
        f"- Session providers: {', '.join(inspection['session_providers'])}",
        "",
        "## Export",
        "",
        _export_section(export),
        "",
        "## Detection agreement",
        "",
        summary["timing_definition"]["matching"],
        "",
        _match_section(detection, "PyTorch/MPS predict()", "ONNX Runtime predict()"),
        "",
        "## Latency",
        "",
        summary["timing_definition"]["elapsed_ms"],
        "",
        summary["timing_definition"]["warmup"],
        "",
        _latency_section("PyTorch/MPS", pytorch),
        "",
        _latency_section("ONNX Runtime CPU", onnx),
        "",
        _speed_conclusion(pytorch, onnx),
        "",
        "## Tracking",
        "",
        _match_section(tracking["boxes"], "PyTorch/MPS track()", "ONNX Runtime track()"),
        "",
        _identity_section(tracking["identity"]),
        "",
        "## Integration",
        "",
        summary.get(
            "pipeline_note",
            "The production command still loads `yolo11n.pt` on MPS. "
            "An ONNX file can be passed through the existing `model_path` "
            "and `--device cpu` settings without an algorithm change. "
            "That path is not the default.",
        ),
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m venv .venv-onnx",
        ".venv-onnx/bin/python -m pip install -r requirements.txt 'onnx>=1.12.0,<2.0.0' onnxruntime",
        ".venv-onnx/bin/python scripts/benchmark_onnx.py --max-frames 100",
        "```",
        "",
        "The exported `yolo11n.onnx` stays in the ignored benchmark directory. "
        "It is not a supported RoadVision backend.",
        "",
    ]
    if "pipeline" in summary:
        lines.extend(["## Pipeline spot check", "", _pipeline_section(summary["pipeline"]), ""])
    return "\n".join(lines)


def compare_prediction_files(reference: Path, candidate: Path) -> dict:
    """Compare pipeline CSVs by key. Categories must be judged explicitly."""

    reference_obs = _read_csv(reference / "observations.csv")
    candidate_obs = _read_csv(candidate / "observations.csv")
    reference_events = _read_csv(reference / "events.csv")
    candidate_events = _read_csv(candidate / "events.csv")
    fields = (
        "class_id",
        "class_name",
        "x1",
        "y1",
        "x2",
        "y2",
        "lane_status",
        "lane_reason",
        "approach_state",
        "evidence",
        "event_state",
    )
    observations = _compare_rows(
        reference_obs,
        candidate_obs,
        ("frame", "track_id"),
        fields,
    )
    events = _compare_rows(
        reference_events,
        candidate_events,
        ("track_id", "start_frame", "end_frame", "confirmation_frame"),
        (
            "duration_frames",
            "positive_frames",
            "interruption_frames",
            "duration_s",
            "min_tau_proxy_s",
        ),
    )
    return {"observations": observations, "events": events}


def _export_section(export: dict) -> str:
    requested = export["requested"]
    inspection = export["inspection"]
    opset = ", ".join(
        f"{item['domain']} {item['version']}" for item in inspection["opsets"]
    )
    inputs = ", ".join(
        f"{item['name']} {item['shape']}" for item in inspection["inputs"]
    )
    outputs = ", ".join(
        f"{item['name']} {item['shape']}" for item in inspection["outputs"]
    )
    nms = (
        "The output rank and width are a raw detection head. "
        "NMS stays in Ultralytics."
        if inspection["external_nms"]
        else "The output shape is consistent with embedded NMS."
    )
    return "\n".join(
        [
            (
                f"Requested format `{requested['format']}`, imgsz {requested['imgsz']}, "
                f"dynamic {requested['dynamic']}, nms {requested['nms']}, "
                f"simplify {requested['simplify']}, quantize {requested['quantize']}, "
                f"opset argument {requested['opset']}."
            ),
            f"Artifact: `{export['artifact']}`",
            f"Bytes: {inspection['bytes']}. SHA-256: `{inspection['sha256']}`",
            f"Opsets: {opset}",
            f"Input: {inputs}",
            f"Output: {outputs}",
            nms,
        ]
    )


def _latency_section(label: str, block: dict) -> str:
    predict = block["predict"]["elapsed"]
    track = block["track"]["elapsed"]
    return "\n".join(
        [
            f"### {label}",
            "",
            f"- Model load: {block['load_ms']:.3f} ms",
            (
                f"- First `predict()`: {block['predict']['warmup_ms']:.3f} ms. "
                f"Steady state over {predict['count']} frames: "
                f"mean {predict['mean_ms']:.3f} ms, median {predict['median_ms']:.3f} ms, "
                f"p95 {predict['p95_ms']:.3f} ms, p99 {predict['p99_ms']:.3f} ms, "
                f"{block['predict']['steady_frames_per_second']:.3f} frames/s."
            ),
            (
                f"- First `track()`: {block['track']['warmup_ms']:.3f} ms. "
                f"Steady state over {track['count']} frames: "
                f"mean {track['mean_ms']:.3f} ms, median {track['median_ms']:.3f} ms, "
                f"p95 {track['p95_ms']:.3f} ms, p99 {track['p99_ms']:.3f} ms, "
                f"{block['track']['steady_frames_per_second']:.3f} frames/s."
            ),
        ]
    )


def _speed_conclusion(pytorch: dict, onnx: dict) -> str:
    pytorch_mean = pytorch["predict"]["elapsed"]["mean_ms"]
    onnx_mean = onnx["predict"]["elapsed"]["mean_ms"]
    if pytorch_mean is None or onnx_mean is None:
        return "Steady-state predict() means were not available."
    delta = onnx_mean - pytorch_mean
    relation = "slower" if delta > 0 else "faster"
    return (
        f"On this Apple M4 sample, ONNX Runtime CPU `predict()` is {relation} "
        f"than synchronized PyTorch/MPS `predict()` by {abs(delta):.3f} ms "
        f"of mean steady-state wall-clock time "
        f"({onnx_mean:.3f} ms versus {pytorch_mean:.3f} ms). "
        "The CPU engine includes the same Ultralytics preprocessing and NMS. "
        "It is the relevant comparison for a Linux host without MPS."
    )


def _match_section(metrics: dict, reference: str, candidate: str) -> str:
    if metrics["matched"] == 0 and metrics["reference_boxes"] == 0:
        return "No boxes were produced."
    minimum = "n/a" if metrics["minimum_iou"] is None else f"{metrics['minimum_iou']:.3f}"
    median = "n/a" if metrics["median_iou"] is None else f"{metrics['median_iou']:.3f}"
    center = (
        "n/a"
        if metrics["median_center_px"] is None
        else f"{metrics['median_center_px']:.3f} px median, {metrics['maximum_center_px']:.3f} px maximum"
    )
    confidence = (
        "n/a"
        if metrics["median_confidence_abs_diff"] is None
        else (
            f"{metrics['median_confidence_abs_diff']:.4f} median absolute, "
            f"{metrics['maximum_confidence_abs_diff']:.4f} maximum"
        )
    )
    return "\n".join(
        [
            f"- {reference}: {metrics['reference_boxes']} boxes",
            f"- {candidate}: {metrics['candidate_boxes']} boxes",
            (
                f"- Matched: {metrics['matched']}. "
                f"Unmatched reference: {metrics['unmatched_reference']}. "
                f"Unmatched candidate: {metrics['unmatched_candidate']}."
            ),
            f"- Matched IoU: median {median}, minimum {minimum}.",
            f"- Center displacement: {center}.",
            f"- Confidence absolute difference: {confidence}.",
            (
                f"- Unmatched boxes at or below confidence {metrics['near_confidence_cutoff']}: "
                f"{metrics['near_threshold_unmatched']}."
            ),
            (
                f"- Matched pairs at or below that confidence with an absolute "
                f"gap of at least {NOTICEABLE_CONFIDENCE_GAP}: "
                f"{metrics['near_threshold_confidence_gaps']}."
            ),
            *(
                [
                    "- Matched boxes whose track IDs agree: "
                    f"{metrics['same_track_id']} of {metrics['track_id_comparisons']}."
                ]
                if metrics.get("track_id_comparisons")
                else []
            ),
        ]
    )


def _identity_section(identity: dict | None) -> str:
    if identity is None:
        return "Track IDs were not present."
    reference = identity["reference"]
    candidate = identity["candidate"]
    return "\n".join(
        [
            "Track IDs are tracker state. A different box score can change "
            "an association without a change to the ByteTrack configuration.",
            "",
            (
                f"- PyTorch IDs: {reference['unique_ids']} unique, "
                f"{reference['fragments']} fragments, "
                f"{reference['ids_with_multiple_fragments']} IDs with a gap."
            ),
            (
                f"- ONNX IDs: {candidate['unique_ids']} unique, "
                f"{candidate['fragments']} fragments, "
                f"{candidate['ids_with_multiple_fragments']} IDs with a gap."
            ),
            (
                f"- Frames whose ID sets differ: "
                f"{identity['frames_with_different_id_sets']}."
            ),
            (
                f"- ID occurrences present on only one side: "
                f"{identity['id_occurrences_only_in_reference']} PyTorch, "
                f"{identity['id_occurrences_only_in_candidate']} ONNX."
            ),
        ]
    )


def _pipeline_section(pipeline: dict) -> str:
    observations = pipeline["observations"]
    events = pipeline["events"]
    lines = [
        (
            f"Observations: {observations['candidate_rows']} ONNX rows, "
            f"{observations['reference_rows']} PyTorch rows, "
            f"{observations['missing']} missing keys, "
            f"{observations['extra']} extra keys, "
            f"{observations['mismatched']} mismatched fields."
        ),
        (
            f"Events: {events['candidate_rows']} ONNX rows, "
            f"{events['reference_rows']} PyTorch rows, "
            f"{events['missing']} missing keys, "
            f"{events['extra']} extra keys, "
            f"{events['mismatched']} mismatched fields."
        ),
    ]
    note = pipeline.get("note")
    if note:
        lines.append(note)
    return "\n".join(lines)


def compare_prediction_rows(reference: list[dict], candidate: list[dict]) -> dict:
    return _compare_rows(
        reference,
        candidate,
        ("frame", "track_id"),
        ("class_name", "lane_status", "approach_state", "evidence", "event_state"),
    )


def _compare_rows(
    reference: list[dict[str, str]],
    candidate: list[dict[str, str]],
    key: tuple[str, ...],
    fields: tuple[str, ...],
) -> dict:
    reference_index = _index(reference, key)
    candidate_index = _index(candidate, key)
    missing = sorted(set(reference_index) - set(candidate_index))
    extra = sorted(set(candidate_index) - set(reference_index))
    mismatched = 0
    for identity in set(reference_index) & set(candidate_index):
        for left, right in zip(reference_index[identity], candidate_index[identity]):
            for field in fields:
                if left.get(field) != right.get(field):
                    mismatched += 1
                    break
    return {
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
        "missing": len(missing),
        "extra": len(extra),
        "mismatched": mismatched,
    }


def _latency_block(rows: list[dict]) -> dict:
    measured = summarize_latency(rows)
    return {
        "load_recorded_separately": True,
        "warmup_ms": measured["warmup_ms"],
        "steady_frames_per_second": measured["steady_frames_per_second"],
        "elapsed": measured["elapsed"],
        "ultralytics_preprocess_ms": measured["ultralytics_preprocess_ms"],
        "ultralytics_inference_ms": measured["ultralytics_inference_ms"],
        "ultralytics_postprocess_ms": measured["ultralytics_postprocess_ms"],
    }


def _write_latency(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LATENCY_COLUMNS)
        writer.writeheader()
        for row in rows:
            stored = {column: row[column] for column in LATENCY_COLUMNS}
            for column, value in stored.items():
                if isinstance(value, float):
                    stored[column] = f"{value:.6f}"
                elif isinstance(value, bool):
                    stored[column] = "true" if value else "false"
                elif value is None:
                    stored[column] = ""
            writer.writerow(stored)


def _write_matches(path: Path, frames: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(frames)


def summary_frames_from_runs(
    reference: list[dict],
    candidate: list[dict],
    call: str,
) -> list[dict]:
    compared = compare_sequences(
        [row["boxes"] for row in reference],
        [row["boxes"] for row in candidate],
    )
    rows = []
    for frame in compared["frames"]:
        rows.append({"call": call, **frame})
    return rows


def _public_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def _tensor_info(value: object) -> dict[str, object]:
    return {"name": value.name, "shape": list(value.shape), "type": value.type}


def _nms_is_external(outputs: list[dict]) -> bool:
    if len(outputs) != 1:
        return False
    shape = outputs[0]["shape"]
    if len(shape) != 3:
        return False
    width = shape[-1]
    anchors = shape[-2]
    if width == 6 and (not isinstance(anchors, int) or anchors <= MAX_DET):
        return False
    return isinstance(width, int) and width > 6


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=float), 50, method="linear")), 6)


def _index(
    rows: list[dict[str, str]],
    key: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    indexed: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        indexed.setdefault(tuple(row[name] for name in key), []).append(row)
    return indexed


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
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


if __name__ == "__main__":
    raise SystemExit(main())
