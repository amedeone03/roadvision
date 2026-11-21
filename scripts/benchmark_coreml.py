"""Compare PyTorch/MPS, ONNX Runtime CPU, and the Core ML execution provider.

The Core ML session is an experiment. It does not change the RoadVision
pipeline. NMS and ByteTrack stay in Ultralytics. The timed calls are
complete Ultralytics predict() or track() wall times. A separate profiled
session records which provider runs each kernel.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATTERN = re.compile(
    r"number of partitions supported by CoreML: (\d+) "
    r"number of nodes in the graph: (\d+) "
    r"number of nodes supported by CoreML: (\d+)"
)
PROVIDER_OPTIONS = {
    "ModelFormat": ("MLProgram", "NeuralNetwork"),
    "MLComputeUnits": ("ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine"),
    "RequireStaticInputShapes": ("0", "1"),
    "EnableOnSubgraphs": ("0", "1"),
    "SpecializationStrategy": ("Default", "FastPrediction"),
    "ProfileComputePlan": ("0", "1"),
    "AllowLowPrecisionAccumulationOnGPU": ("0", "1"),
}
COREML_OPTIONS = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
    "RequireStaticInputShapes": "1",
    "EnableOnSubgraphs": "1",
    "SpecializationStrategy": "FastPrediction",
    "ProfileComputePlan": "0",
    "AllowLowPrecisionAccumulationOnGPU": "0",
}
NEURAL_NETWORK_OPTIONS = {
    "ModelFormat": "NeuralNetwork",
    "MLComputeUnits": "ALL",
    "RequireStaticInputShapes": "1",
    "EnableOnSubgraphs": "1",
    "SpecializationStrategy": "Default",
    "ProfileComputePlan": "0",
    "AllowLowPrecisionAccumulationOnGPU": "0",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/videos/test_drive.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--onnx", type=Path, default=ROOT / "runs/onnx_benchmark/yolo11n.onnx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/coreml_benchmark")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--skip-pipeline", action="store_true")
    args = parser.parse_args(argv)

    benchmark = _onnx_benchmark()
    if not benchmark.probe_onnx_runtime()["available"]:
        print("ONNX Runtime is not installed. Use .venv-onnx.", file=sys.stderr)
        return 2
    problems = validate_provider_options(COREML_OPTIONS)
    if problems or args.max_frames < 2:
        print(f"Invalid Core ML configuration: {problems or 'max-frames'}", file=sys.stderr)
        return 2
    if not args.onnx.is_file():
        print(f"Exported model not found: {args.onnx}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = benchmark.RoadVisionConfig()
    frames = benchmark.decode_frames(args.video, args.max_frames)
    if len(frames) < 2:
        print("Need at least two decoded frames.", file=sys.stderr)
        return 1

    inspection = benchmark.inspect_onnx_artifact(args.onnx)
    if not inspection["external_nms"]:
        print("The ONNX model does not keep NMS outside the graph.", file=sys.stderr)
        return 1
    primary = inspect_provider(args.onnx, COREML_OPTIONS, args.output_dir / "mlprogram")
    neural = inspect_provider(args.onnx, NEURAL_NETWORK_OPTIONS, args.output_dir / "neuralnetwork")

    common = benchmark.prediction_kwargs(config)
    providers = [("CoreMLExecutionProvider", COREML_OPTIONS), "CPUExecutionProvider"]
    pytorch_predict, pytorch_load = benchmark.time_call_sequence(
        args.weights, frames, {**common, "device": "mps"}, call="predict", engine="pytorch_mps"
    )
    cpu_predict, cpu_load = benchmark.time_call_sequence(
        args.onnx, frames, {**common, "device": "cpu"}, call="predict", engine="onnx_cpu"
    )
    coreml_predict, coreml_load = benchmark.time_call_sequence(
        args.onnx,
        frames,
        {**common, "device": "cpu"},
        call="predict",
        engine="onnx_coreml",
        providers=providers,
    )
    track_kwargs = {"persist": True, "tracker": config.tracker.config_path}
    pytorch_track, _ = benchmark.time_call_sequence(
        args.weights,
        frames,
        {**common, "device": "mps", **track_kwargs},
        call="track",
        engine="pytorch_mps",
    )
    cpu_track, _ = benchmark.time_call_sequence(
        args.onnx,
        frames,
        {**common, "device": "cpu", **track_kwargs},
        call="track",
        engine="onnx_cpu",
    )
    coreml_track, _ = benchmark.time_call_sequence(
        args.onnx,
        frames,
        {**common, "device": "cpu", **track_kwargs},
        call="track",
        engine="onnx_coreml",
        providers=providers,
    )

    summary = {
        "artifact": inspection,
        "provider": {
            "requested": COREML_OPTIONS,
            "mlprogram": primary,
            "neuralnetwork": _public_assignment(neural),
            "timed_session_providers": coreml_predict[0].get("session_providers", []),
        },
        "latency": {
            "pytorch_mps": {
                "load_ms": pytorch_load,
                "predict": benchmark.summarize_latency(pytorch_predict),
                "track": benchmark.summarize_latency(pytorch_track),
            },
            "onnx_cpu": {
                "load_ms": cpu_load,
                "predict": benchmark.summarize_latency(cpu_predict),
                "track": benchmark.summarize_latency(cpu_track),
            },
            "onnx_coreml": {
                "load_ms": coreml_load,
                "predict": benchmark.summarize_latency(coreml_predict),
                "track": benchmark.summarize_latency(coreml_track),
            },
        },
        "detection": {
            "versus_pytorch": benchmark.compare_sequences(
                [row["boxes"] for row in pytorch_predict],
                [row["boxes"] for row in coreml_predict],
            )["detection"],
            "versus_onnx_cpu": benchmark.compare_sequences(
                [row["boxes"] for row in cpu_predict],
                [row["boxes"] for row in coreml_predict],
            )["detection"],
        },
        "tracking": {
            "versus_pytorch": _tracking_summary(benchmark, pytorch_track, coreml_track),
            "versus_onnx_cpu": _tracking_summary(benchmark, cpu_track, coreml_track),
        },
    }
    if not args.skip_pipeline:
        summary["pipeline"] = run_pipeline_spot_check(
            benchmark, args.video, args.onnx, args.output_dir / "pipeline", args.max_frames, providers
        )
    (args.output_dir / "coreml_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = render_report(summary)
    (args.output_dir / "coreml_report.md").write_text(report)
    print(f"Wrote {args.output_dir / 'coreml_report.md'}")
    return 0


def validate_provider_options(options: dict) -> list[str]:
    """Return option names this ONNX Runtime build does not accept."""

    problems = []
    for name, value in options.items():
        allowed = PROVIDER_OPTIONS.get(name)
        if allowed is None or str(value) not in allowed:
            problems.append(name)
    return problems


def parse_coreml_capability(log: str) -> dict | None:
    """Read the partition line ONNX Runtime prints while assigning nodes."""

    match = CAPABILITY_PATTERN.search(log)
    if match is None:
        return None
    partitions, nodes, supported = (int(item) for item in match.groups())
    return {
        "partitions": partitions,
        "graph_nodes": nodes,
        "supported_nodes": supported,
        "unsupported_nodes": nodes - supported,
    }


def support_notes(log: str) -> list[str]:
    """Keep unique lines that name an operator the provider declined."""

    notes = []
    for line in log.splitlines():
        if "not supported" not in line.lower() and "Skipping" not in line:
            continue
        text = line.split("] ", 1)[-1].strip()
        if text and text not in notes:
            notes.append(text)
    return notes[:20]


def summarize_ort_profile(events: list[dict]) -> dict:
    """Count each profiled kernel once, even when several runs repeat it."""

    kernels: dict[str, dict] = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        name = str(event.get("name", ""))
        if name.endswith("_kernel_time"):
            name = name[: -len("_kernel_time")]
        kernels.setdefault(name, event.get("args") or {})
    by_provider: dict[str, int] = {}
    cpu_ops: Counter[str] = Counter()
    for args in kernels.values():
        provider = str(args.get("provider", "unknown"))
        by_provider[provider] = by_provider.get(provider, 0) + 1
        if provider != "CoreMLExecutionProvider":
            cpu_ops[str(args.get("op_name", "unknown"))] += 1
    return {
        "kernels_by_provider": by_provider,
        "cpu_operators": dict(cpu_ops),
        "coreml_kernels": by_provider.get("CoreMLExecutionProvider", 0),
        "cpu_kernels": sum(
            count for provider, count in by_provider.items() if provider != "CoreMLExecutionProvider"
        ),
    }


def match_trajectories(
    reference_rows: list[dict],
    candidate_rows: list[dict],
    *,
    min_iou: float = 0.5,
) -> dict:
    """Map tracks by overlapping boxes. Equal numeric IDs are not required."""

    benchmark = _onnx_benchmark()
    reference = _tracks(reference_rows)
    candidate_frames = _boxes_by_frame(candidate_rows)
    overlaps = []
    for reference_id, boxes in reference.items():
        by_candidate: dict[int, list[float]] = {}
        for box in boxes:
            for candidate_id, peer in candidate_frames.get(box["frame"], []):
                if peer["cls"] != box["cls"]:
                    continue
                value = benchmark.box_iou(box["xyxy"], peer["xyxy"])
                if value >= min_iou:
                    by_candidate.setdefault(candidate_id, []).append(value)
        for candidate_id, values in by_candidate.items():
            overlaps.append((reference_id, candidate_id, len(values), sum(values) / len(values)))

    fragmented = sorted(
        reference_id
        for reference_id in reference
        if sum(1 for item in overlaps if item[0] == reference_id) >= 2
    )
    pairs = []
    used_reference, used_candidate = set(), set()
    for reference_id, candidate_id, shared, mean_iou in sorted(
        overlaps, key=lambda item: (-item[2], -item[3], item[0], item[1])
    ):
        if reference_id in used_reference or candidate_id in used_candidate:
            continue
        used_reference.add(reference_id)
        used_candidate.add(candidate_id)
        switches = 0
        compared = 0
        for box in reference[reference_id]:
            scored = []
            for other_id, peer in candidate_frames.get(box["frame"], []):
                if peer["cls"] != box["cls"]:
                    continue
                value = benchmark.box_iou(box["xyxy"], peer["xyxy"])
                if value >= min_iou:
                    scored.append((value, other_id))
            if not scored:
                continue
            compared += 1
            if max(scored)[1] != candidate_id:
                switches += 1
        pairs.append(
            {
                "reference_id": reference_id,
                "candidate_id": candidate_id,
                "shared_frames": shared,
                "mean_iou": round(mean_iou, 6),
                "same_id": reference_id == candidate_id,
                "identity_switches": switches,
                "frames_compared": compared,
            }
        )
    candidate_count = len(_tracks(candidate_rows))
    return {
        "reference_tracks": len(reference),
        "candidate_tracks": candidate_count,
        "mapped": len(pairs),
        "same_id": sum(pair["same_id"] for pair in pairs),
        "renumbered": sum(not pair["same_id"] for pair in pairs),
        "identity_switches": sum(pair["identity_switches"] for pair in pairs),
        "fragmented_reference_tracks": len(fragmented),
        "unmatched_reference": len(reference) - len(pairs),
        "unmatched_candidate": candidate_count - len(pairs),
        "pairs": pairs,
    }


def match_episodes(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    """Match evidence episodes by frame overlap, ignoring track IDs."""

    proposals = []
    for ref_index, reference in enumerate(reference_rows):
        for cand_index, candidate in enumerate(candidate_rows):
            overlap = _temporal_iou(reference, candidate)
            if overlap > 0:
                proposals.append((-overlap, ref_index, cand_index, overlap))
    used_reference, used_candidate = set(), set()
    pairs = []
    for _, ref_index, cand_index, overlap in sorted(proposals):
        if ref_index in used_reference or cand_index in used_candidate:
            continue
        used_reference.add(ref_index)
        used_candidate.add(cand_index)
        reference = reference_rows[ref_index]
        candidate = candidate_rows[cand_index]
        pairs.append(
            {
                "reference_track_id": reference.get("track_id"),
                "candidate_track_id": candidate.get("track_id"),
                "reference_window": [int(reference["start_frame"]), int(reference["end_frame"])],
                "candidate_window": [int(candidate["start_frame"]), int(candidate["end_frame"])],
                "temporal_iou": round(overlap, 6),
                "same_bounds": (
                    reference["start_frame"] == candidate["start_frame"]
                    and reference["end_frame"] == candidate["end_frame"]
                ),
            }
        )
    return {
        "reference_episodes": len(reference_rows),
        "candidate_episodes": len(candidate_rows),
        "mapped": len(pairs),
        "same_bounds": sum(pair["same_bounds"] for pair in pairs),
        "unmatched_reference": len(reference_rows) - len(pairs),
        "unmatched_candidate": len(candidate_rows) - len(pairs),
        "pairs": pairs,
    }


def inspect_provider(model: Path, options: dict, prefix: Path) -> dict:
    """Load one profiled session and record the provider that owns each kernel."""

    import numpy as np
    import onnxruntime

    prefix.parent.mkdir(parents=True, exist_ok=True)
    session_options = onnxruntime.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(prefix)
    session_options.log_severity_level = 1
    requested = [("CoreMLExecutionProvider", dict(options)), "CPUExecutionProvider"]

    def create():
        return onnxruntime.InferenceSession(
            str(model), session_options, providers=requested
        )

    session, create_log = _capture_stderr(create)
    model_input = session.get_inputs()[0]
    shape = [int(dim) for dim in model_input.shape]
    zeros = np.zeros(shape, dtype=np.float32)

    def execute():
        session.run(None, {model_input.name: zeros})

    _, run_log = _capture_stderr(execute)
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text())
    echoed = dict(session.get_provider_options().get("CoreMLExecutionProvider", {}))
    return {
        "requested_options": dict(options),
        "effective_options": echoed,
        "session_providers": list(session.get_providers()),
        "input_shape": shape,
        "static_input": all(isinstance(dim, int) and dim > 0 for dim in model_input.shape),
        "capability": parse_coreml_capability(create_log + "\n" + run_log),
        "kernels": summarize_ort_profile(events),
        "support_notes": support_notes(create_log + "\n" + run_log),
        "profile_path": str(profile_path),
    }


def run_pipeline_spot_check(benchmark, video: Path, artifact: Path, output_dir: Path, frames: int, providers: list) -> dict:
    """Run 100 frames through the unchanged pipeline with a forced Core ML session."""

    from dataclasses import replace

    from roadvision.config import load_config

    config = load_config(ROOT / "configs/roadvision.yaml")
    config = replace(
        config,
        video=replace(
            config.video,
            input_path=str(video),
            output_dir=str(output_dir),
            max_frames=frames,
        ),
        detector=replace(config.detector, model_path=str(artifact), device="cpu"),
    )
    from roadvision.pipeline import run_video

    with _forced_sessions(providers):
        result = run_video(config)
    reference = ROOT / "runs/onnx_benchmark/pytorch_pipeline"
    compared = benchmark.compare_prediction_files(reference, output_dir)
    episodes = match_episodes(
        benchmark._read_csv(reference / "events.csv"),
        benchmark._read_csv(output_dir / "events.csv"),
    )
    return {
        "frames_processed": result.frames_processed,
        "observations": result.observation_count,
        "episodes": result.episode_count,
        "elapsed_s": result.elapsed_seconds,
        "strict_keys": compared,
        "episode_windows": episodes,
        "note": (
            "The production CLI cannot select these Core ML options. "
            "This run forced the session providers around the unchanged pipeline. "
            "The PyTorch reference is runs/onnx_benchmark/pytorch_pipeline."
        ),
    }


def render_report(summary: dict) -> str:
    provider = summary["provider"]
    primary = provider["mlprogram"]
    kernels = primary["kernels"]
    capability = primary.get("capability") or {}
    lines = [
        "# ONNX Runtime Core ML provider benchmark",
        "",
        "Experimental only. PyTorch/MPS remains the default. "
        "Timed values are wall clock around one Ultralytics call, including preprocessing and Python NMS. "
        "PyTorch/MPS synchronizes around the call. ONNX calls use ordinary wall-clock time. "
        "The first frame of each fresh model is warm-up and includes Core ML compilation when it happens there.",
        "",
        "## Provider configuration",
        "",
        _option_lines(provider["requested"]),
        f"- Timed session providers: {', '.join(provider.get('timed_session_providers') or [])}.",
        f"- CPUExecutionProvider stays registered as the fallback.",
        f"- Static input shape: {primary.get('input_shape')}.",
        "",
        "## Node execution",
        "",
        (
            f"MLProgram assigned {capability.get('supported_nodes', 'unknown')} of "
            f"{capability.get('graph_nodes', 'unknown')} graph nodes across "
            f"{capability.get('partitions', 'unknown')} Core ML partition(s)."
        ),
        (
            f"Profiled kernels: {kernels.get('coreml_kernels', 0)} Core ML, "
            f"{kernels.get('cpu_kernels', 0)} CPU."
        ),
        _fallback_sentence(kernels),
        _neural_sentence(provider.get("neuralnetwork")),
        "",
        "## Latency",
        "",
        _latency_block("PyTorch/MPS", summary["latency"]["pytorch_mps"]),
        _latency_block("ONNX Runtime CPU", summary["latency"]["onnx_cpu"]),
        _latency_block("ONNX Runtime Core ML", summary["latency"]["onnx_coreml"]),
        "",
        _speed_sentence(summary["latency"]),
        "",
        "## Detections",
        "",
        "Matching is greedy, one-to-one, within class, at IoU >= 0.5, before tracking.",
        "",
        _detection_block("Core ML versus PyTorch/MPS", summary["detection"]["versus_pytorch"]),
        "",
        _detection_block("Core ML versus ONNX Runtime CPU", summary["detection"]["versus_onnx_cpu"]),
        "",
        "## Tracking",
        "",
        "Trajectories are mapped by box overlap. A different numeric ID with the same boxes is renumbering.",
        "",
        _tracking_block("Core ML versus PyTorch/MPS", summary["tracking"]["versus_pytorch"]),
        "",
        _tracking_block("Core ML versus ONNX Runtime CPU", summary["tracking"]["versus_onnx_cpu"]),
    ]
    if "pipeline" in summary:
        lines.extend(["", "## Pipeline spot check", "", _pipeline_block(summary["pipeline"])])
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            ".venv-onnx/bin/python scripts/benchmark_coreml.py --max-frames 100",
            "```",
            "",
            "The exported model and this output stay in ignored `runs/` directories.",
            "",
        ]
    )
    return "\n".join(lines)


def _tracking_summary(benchmark, reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    compared = benchmark.compare_sequences(
        [row["boxes"] for row in reference_rows],
        [row["boxes"] for row in candidate_rows],
    )
    return {
        "boxes": compared["detection"],
        "trajectories": match_trajectories(reference_rows, candidate_rows),
    }


def _tracks(rows: list[dict]) -> dict[int, list[dict]]:
    tracks: dict[int, list[dict]] = {}
    for row in rows:
        for box in row["boxes"]:
            if "track_id" not in box:
                continue
            tracks.setdefault(int(box["track_id"]), []).append({"frame": int(row["frame"]), **box})
    return tracks


def _boxes_by_frame(rows: list[dict]) -> dict[int, list[tuple[int, dict]]]:
    frames: dict[int, list[tuple[int, dict]]] = {}
    for track_id, boxes in _tracks(rows).items():
        for box in boxes:
            frames.setdefault(box["frame"], []).append((track_id, box))
    return frames


def _temporal_iou(reference: dict, candidate: dict) -> float:
    start = max(int(reference["start_frame"]), int(candidate["start_frame"]))
    end = min(int(reference["end_frame"]), int(candidate["end_frame"]))
    overlap = max(0, end - start + 1)
    union = (
        int(reference["end_frame"]) - int(reference["start_frame"]) + 1
        + int(candidate["end_frame"]) - int(candidate["start_frame"]) + 1
        - overlap
    )
    if union <= 0:
        return 0.0
    return overlap / union


def _public_assignment(inspection: dict) -> dict:
    return {
        "effective_options": inspection.get("effective_options"),
        "capability": inspection.get("capability"),
        "kernels": inspection.get("kernels"),
        "support_notes": inspection.get("support_notes"),
    }


def _fallback_sentence(kernels: dict) -> str:
    if kernels.get("cpu_kernels", 0) == 0 and kernels.get("coreml_kernels", 0) > 0:
        return (
            "The profiled MLProgram run executed its kernels on Core ML. "
            "CPUExecutionProvider was available and received no kernels."
        )
    if kernels.get("coreml_kernels", 0) == 0:
        return "The profiled run executed no Core ML kernels. The model ran on CPU."
    operators = ", ".join(
        f"{name} x{count}" for name, count in sorted((kernels.get("cpu_operators") or {}).items())
    )
    return f"CPU fallback kernels: {kernels.get('cpu_kernels', 0)} ({operators})."


def _neural_sentence(neural: dict | None) -> str:
    if not neural:
        return "The NeuralNetwork format was not inspected."
    capability = neural.get("capability") or {}
    kernels = neural.get("kernels") or {}
    operators = ", ".join(
        f"{name} x{count}" for name, count in sorted((kernels.get("cpu_operators") or {}).items())
    ) or "none"
    return (
        "The NeuralNetwork format, which is what Ultralytics selects from its static-shape flag, "
        f"supported {capability.get('supported_nodes', 'unknown')} of "
        f"{capability.get('graph_nodes', 'unknown')} nodes. "
        f"Its CPU operators were {operators}. The timed session uses MLProgram instead."
    )


def _option_lines(options: dict) -> str:
    return "\n".join(f"- `{name}`: `{value}`" for name, value in options.items())


def _latency_block(title: str, block: dict) -> str:
    predict = block["predict"]["elapsed"]
    track = block["track"]["elapsed"]
    return "\n".join(
        [
            f"### {title}",
            "",
            f"- Model load: {block['load_ms']:.3f} ms.",
            (
                f"- First predict: {block['predict']['warmup_ms']:.3f} ms. "
                f"Steady mean / median / p95 / p99: "
                f"{predict['mean_ms']:.3f} / {predict['median_ms']:.3f} / "
                f"{predict['p95_ms']:.3f} / {predict['p99_ms']:.3f} ms. "
                f"Throughput {block['predict']['steady_frames_per_second']:.3f} frames/s."
            ),
            (
                f"- Steady track mean: {track['mean_ms']:.3f} ms. "
                f"Throughput {block['track']['steady_frames_per_second']:.3f} frames/s."
            ),
        ]
    )


def _speed_sentence(latency: dict) -> str:
    core = latency["onnx_coreml"]["predict"]["elapsed"]["mean_ms"]
    pytorch = latency["pytorch_mps"]["predict"]["elapsed"]["mean_ms"]
    cpu = latency["onnx_cpu"]["predict"]["elapsed"]["mean_ms"]
    relation = "faster" if core < pytorch else "slower"
    return (
        f"Core ML predict() is {relation} than synchronized PyTorch/MPS by "
        f"{abs(core - pytorch):.3f} ms of mean steady-state time "
        f"({core:.3f} ms versus {pytorch:.3f} ms). "
        f"ONNX Runtime CPU mean predict() is {cpu:.3f} ms. "
        "The Core ML figure includes Ultralytics preprocessing and NMS, not only the provider kernel."
    )


def _detection_block(title: str, metrics: dict) -> str:
    return "\n".join(
        [
            f"### {title}",
            "",
            (
                f"{metrics['candidate_boxes']} Core ML boxes, {metrics['reference_boxes']} reference boxes, "
                f"{metrics['matched']} matched, {metrics['unmatched_reference']} unmatched reference, "
                f"{metrics['unmatched_candidate']} unmatched candidate."
            ),
            (
                f"Median IoU {_fmt(metrics['median_iou'])}, minimum IoU {_fmt(metrics['minimum_iou'])}. "
                f"Median center shift {_fmt(metrics['median_center_px'])} px, "
                f"maximum {_fmt(metrics['maximum_center_px'])} px."
            ),
            (
                f"Median absolute confidence gap {_fmt(metrics['median_confidence_abs_diff'])}, "
                f"maximum {_fmt(metrics['maximum_confidence_abs_diff'])}. "
                f"Unmatched boxes at or below confidence {metrics['near_confidence_cutoff']}: "
                f"{metrics['near_threshold_unmatched']}. "
                f"Matched pairs near that confidence with a gap of at least 0.02: "
                f"{metrics['near_threshold_confidence_gaps']}."
            ),
        ]
    )


def _tracking_block(title: str, tracking: dict) -> str:
    boxes = tracking["boxes"]
    trajectories = tracking["trajectories"]
    return "\n".join(
        [
            f"### {title}",
            "",
            (
                f"Boxes: {boxes['matched']} matched of {boxes['reference_boxes']} reference and "
                f"{boxes['candidate_boxes']} Core ML. Median IoU {_fmt(boxes['median_iou'])}."
            ),
            (
                f"Trajectories: {trajectories['mapped']} mapped of {trajectories['reference_tracks']} reference and "
                f"{trajectories['candidate_tracks']} Core ML. "
                f"Same ID {trajectories['same_id']}. Renumbered {trajectories['renumbered']}. "
                f"Fragmented reference tracks {trajectories['fragmented_reference_tracks']}. "
                f"Identity-switch frames inside mapped tracks {trajectories['identity_switches']}. "
                f"Unmatched reference {trajectories['unmatched_reference']}, "
                f"unmatched candidate {trajectories['unmatched_candidate']}."
            ),
        ]
    )


def _pipeline_block(pipeline: dict) -> str:
    observations = pipeline["strict_keys"]["observations"]
    events = pipeline["strict_keys"]["events"]
    windows = pipeline["episode_windows"]
    elapsed = pipeline.get("elapsed_s")
    elapsed_text = "unknown" if elapsed is None else f"{elapsed:.3f} s"
    return "\n".join(
        [
            pipeline["note"],
            (
                f"Processed {pipeline['frames_processed']} frames, "
                f"{pipeline['observations']} observations, {pipeline['episodes']} episodes "
                f"in {elapsed_text}."
            ),
            (
                f"Strict observation keys: {observations['missing']} missing, "
                f"{observations['extra']} extra, {observations['mismatched']} mismatched."
            ),
            (
                f"Episode windows: {windows['mapped']} mapped, {windows['same_bounds']} with the same bounds, "
                f"{windows['unmatched_reference']} unmatched reference, "
                f"{windows['unmatched_candidate']} unmatched candidate."
            ),
            (
                f"Strict event keys: {events['missing']} missing, {events['extra']} extra, "
                f"{events['mismatched']} mismatched."
            ),
        ]
    )


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _capture_stderr(function):
    saved = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    chunks: list[bytes] = []

    def read() -> None:
        while True:
            data = os.read(read_fd, 65536)
            if not data:
                break
            chunks.append(data)

    thread = threading.Thread(target=read)
    thread.start()
    try:
        result = function()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(write_fd)
        thread.join()
        os.close(read_fd)
    return result, b"".join(chunks).decode("utf-8", "replace")


class _forced_sessions:
    def __init__(self, providers: list) -> None:
        self.providers = list(providers)
        self.original = None

    def __enter__(self):
        import onnxruntime

        self.original = onnxruntime.InferenceSession
        forced = self.providers

        def session(model, sess_options=None, providers=None, *args, **kwargs):
            return self.original(model, sess_options, providers=forced, *args, **kwargs)

        onnxruntime.InferenceSession = session
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        import onnxruntime

        onnxruntime.InferenceSession = self.original


def _onnx_benchmark():
    module = sys.modules.get("benchmark_onnx")
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location("benchmark_onnx", ROOT / "scripts" / "benchmark_onnx.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_onnx"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
