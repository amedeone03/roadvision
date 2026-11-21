"""Full-video Core ML validation. The production CLI is not involved.

The provider override stays inside this process and is restored before
the function returns, including when the run raises. Comparison helpers
do not import ONNX Runtime.
"""

import argparse
import json
import logging
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ELAPSED_SECONDS = 49.462231
REFERENCE_FPS = 34.935747
CATEGORY_FIELDS = ("lane_status", "approach_state", "evidence", "event_state")
EPISODE_FIELDS = (
    "confirmation_frame",
    "duration_frames",
    "positive_frames",
    "interruption_frames",
    "min_tau_proxy_s",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/videos/test_drive.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--onnx", type=Path, default=ROOT / "runs/onnx_benchmark/yolo11n.onnx")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/full_regression")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/coreml_full_validation")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.run:
        run_validation(args.video, args.weights, args.onnx, args.output_dir)
    reference = args.reference
    coreml_dir = args.output_dir / "coreml"
    pytorch_dir = args.output_dir / "pytorch_mps"
    if not (coreml_dir / "observations.csv").is_file():
        print(f"Core ML outputs not found: {coreml_dir}", file=sys.stderr)
        return 1
    summary = {
        "historical_unprofiled_baseline": {
            "directory": str(reference),
            "elapsed_seconds": REFERENCE_ELAPSED_SECONDS,
            "frames_per_second": REFERENCE_FPS,
            "paired": False,
            "note": (
                "Published PyTorch/MPS full-video run. It is not a paired "
                "measurement with the Core ML run."
            ),
        },
        "coreml_versus_historical": compare_directories(reference, coreml_dir),
        "runs": {
            "coreml": _run_record(coreml_dir),
            "coreml_profile": _profile_record(args.output_dir / "coreml_profile"),
        },
    }
    if (pytorch_dir / "observations.csv").is_file():
        summary["runs"]["pytorch_mps"] = _run_record(pytorch_dir)
        summary["coreml_versus_fresh_pytorch"] = compare_directories(pytorch_dir, coreml_dir)
    report = render_report(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "validation_report.md").write_text(report)
    print(f"Wrote {args.output_dir / 'validation_report.md'}")
    return 0


def run_validation(video: Path, weights: Path, artifact: Path, output_dir: Path) -> None:
    """Run one unprofiled Core ML pass, one fresh PyTorch pass, and one profiled Core ML pass."""

    coreml = _coreml()
    benchmark = coreml._onnx_benchmark()
    if not benchmark.probe_onnx_runtime()["available"]:
        raise RuntimeError("ONNX Runtime is not installed. Use .venv-onnx.")
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection = coreml.inspect_provider(artifact, coreml.COREML_OPTIONS, output_dir / "provider_probe")
    (output_dir / "provider_inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")
    providers = [("CoreMLExecutionProvider", coreml.COREML_OPTIONS), "CPUExecutionProvider"]
    _run_pipeline(
        video,
        output_dir / "coreml",
        model_path=artifact,
        device="cpu",
        providers=providers,
        profile=False,
    )
    _run_pipeline(
        video,
        output_dir / "pytorch_mps",
        model_path=weights,
        device="mps",
        providers=None,
        profile=False,
    )
    _run_pipeline(
        video,
        output_dir / "coreml_profile",
        model_path=artifact,
        device="cpu",
        providers=providers,
        profile=True,
    )


def compare_directories(reference: Path, candidate: Path) -> dict:
    """Compare observation and episode CSVs. Track IDs are not an identity key."""

    benchmark = _coreml()._onnx_benchmark()
    reference_rows = benchmark._read_csv(reference / "observations.csv")
    candidate_rows = benchmark._read_csv(candidate / "observations.csv")
    reference_events = benchmark._read_csv(reference / "events.csv")
    candidate_events = benchmark._read_csv(candidate / "events.csv")
    frame_count = _frame_count(reference, candidate)
    return {
        "counts": frame_count_delta(reference_rows, candidate_rows, frame_count),
        "classes": {
            "reference": dict(Counter(row["class_name"] for row in reference_rows)),
            "candidate": dict(Counter(row["class_name"] for row in candidate_rows)),
        },
        "detections": detection_agreement(reference_rows, candidate_rows),
        "tracking": tracking_agreement(reference_rows, candidate_rows),
        "categories": category_agreement(reference_rows, candidate_rows),
        "episodes": episode_agreement(reference_events, candidate_events),
    }


def frame_count_delta(reference_rows: list[dict], candidate_rows: list[dict], frame_count: int) -> dict:
    reference = Counter(int(row["frame"]) for row in reference_rows)
    candidate = Counter(int(row["frame"]) for row in candidate_rows)
    deltas = [candidate[frame] - reference[frame] for frame in range(1, frame_count + 1)]
    return {
        "frames": frame_count,
        "reference_observations": len(reference_rows),
        "candidate_observations": len(candidate_rows),
        "frames_with_different_counts": sum(delta != 0 for delta in deltas),
        "max_abs_count_delta": max((abs(delta) for delta in deltas), default=0),
    }


def detection_agreement(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    benchmark = _coreml()._onnx_benchmark()
    reference = _boxes_by_frame(reference_rows)
    candidate = _boxes_by_frame(candidate_rows)
    frames = sorted(set(reference) | set(candidate))
    matched = [
        benchmark.match_boxes(reference.get(frame, []), candidate.get(frame, []))
        for frame in frames
    ]
    summary = benchmark.summarize_matches(matched)
    ious = [pair["iou"] for frame in matched for pair in frame["pairs"]]
    summary["p05_iou"] = _percentile(ious, 5)
    gaps = [pair["confidence_abs_diff"] for frame in matched for pair in frame["pairs"]]
    centers = [pair["center_px"] for frame in matched for pair in frame["pairs"]]
    summary["p95_center_px"] = _percentile(centers, 95)
    summary["p95_confidence_abs_diff"] = _percentile(gaps, 95)
    return summary


def category_agreement(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    benchmark = _coreml()._onnx_benchmark()
    reference = _boxes_by_frame(reference_rows)
    candidate = _boxes_by_frame(candidate_rows)
    frames = sorted(set(reference) | set(candidate))
    pairs = [
        pair
        for frame in frames
        for pair in benchmark.match_boxes(reference.get(frame, []), candidate.get(frame, []))["pairs"]
    ]
    categories = {}
    for field in CATEGORY_FIELDS:
        mismatches = [
            pair for pair in pairs
            if pair.get(f"reference_{field}") != pair.get(f"candidate_{field}")
        ]
        categories[field] = {"compared": len(pairs), "mismatched": len(mismatches)}
    tau_gaps = []
    tau_missing = 0
    for pair in pairs:
        left = pair.get("reference_tau_proxy_s")
        right = pair.get("candidate_tau_proxy_s")
        if left is None or right is None:
            tau_missing += 1
            continue
        tau_gaps.append(abs(float(left) - float(right)))
    categories["tau_proxy_s"] = {
        "compared": len(tau_gaps),
        "missing_on_one_side": tau_missing,
        "median_abs_diff": _percentile(tau_gaps, 50),
        "p95_abs_diff": _percentile(tau_gaps, 95),
        "maximum_abs_diff": max(tau_gaps) if tau_gaps else None,
    }
    return categories


def tracking_agreement(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    coreml = _coreml()
    reference_tracks = _track_rows(reference_rows)
    candidate_tracks = _track_rows(candidate_rows)
    mapped = coreml.match_trajectories(reference_tracks, candidate_tracks)
    reference_lengths = _track_lengths(reference_rows)
    candidate_lengths = _track_lengths(candidate_rows)
    deltas = []
    for pair in mapped["pairs"]:
        delta = candidate_lengths[pair["candidate_id"]] - reference_lengths[pair["reference_id"]]
        pair["duration_frame_delta"] = delta
        deltas.append(delta)
    mapped["duration"] = {
        "median_abs_frame_delta": _percentile([abs(delta) for delta in deltas], 50),
        "pairs_differing_by_at_least_5": sum(abs(delta) >= 5 for delta in deltas),
        "max_abs_frame_delta": max((abs(delta) for delta in deltas), default=0),
    }
    return mapped


def episode_agreement(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    coreml = _coreml()
    mapped = coreml.match_episodes(reference_rows, candidate_rows)
    details = []
    for pair in mapped["pairs"]:
        reference = _episode_row(reference_rows, pair["reference_track_id"], pair["reference_window"])
        candidate = _episode_row(candidate_rows, pair["candidate_track_id"], pair["candidate_window"])
        fields = {}
        for field in EPISODE_FIELDS:
            fields[field] = {
                "reference": reference.get(field),
                "candidate": candidate.get(field),
                "equal": reference.get(field) == candidate.get(field),
            }
        details.append({**pair, "fields": fields})
    used_reference = {
        (str(pair["reference_track_id"]), tuple(pair["reference_window"])) for pair in mapped["pairs"]
    }
    used_candidate = {
        (str(pair["candidate_track_id"]), tuple(pair["candidate_window"])) for pair in mapped["pairs"]
    }
    return {
        "reference_episodes": mapped["reference_episodes"],
        "candidate_episodes": mapped["candidate_episodes"],
        "mapped": mapped["mapped"],
        "same_bounds": mapped["same_bounds"],
        "pairs": details,
        "unmatched_reference": [
            row for row in reference_rows
            if (str(row["track_id"]), (int(row["start_frame"]), int(row["end_frame"]))) not in used_reference
        ],
        "unmatched_candidate": [
            row for row in candidate_rows
            if (str(row["track_id"]), (int(row["start_frame"]), int(row["end_frame"]))) not in used_candidate
        ],
    }


def _run_pipeline(video, output_dir, *, model_path, device, providers, profile: bool) -> None:
    from dataclasses import replace

    from roadvision.config import load_config
    from roadvision.pipeline import run_video

    config = load_config(ROOT / "configs/roadvision.yaml")
    config = replace(
        config,
        video=replace(config.video, input_path=str(video), output_dir=str(output_dir), max_frames=None),
        detector=replace(config.detector, model_path=str(model_path), device=device),
    )
    logging.info("Starting %s on %s profile=%s", output_dir, device, profile)
    if providers is None:
        result = run_video(config, profile=profile)
    else:
        import onnxruntime

        original = onnxruntime.InferenceSession
        with _coreml()._forced_sessions(providers):
            result = run_video(config, profile=profile)
        if onnxruntime.InferenceSession is not original:
            raise RuntimeError("The Core ML provider override leaked after the run.")
    logging.info(
        "Finished %s: %s frames, %s observations, %s episodes, %.3f s",
        output_dir,
        result.frames_processed,
        result.observation_count,
        result.episode_count,
        result.elapsed_seconds,
    )


def _boxes_by_frame(rows: list[dict]) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}
    for row in rows:
        tau = row.get("tau_proxy_s", "")
        frames.setdefault(int(row["frame"]), []).append(
            {
                "cls": int(row["class_id"]),
                "conf": float(row["confidence"]),
                "xyxy": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                "track_id": int(row["track_id"]),
                "lane_status": row["lane_status"],
                "approach_state": row["approach_state"],
                "evidence": row["evidence"],
                "event_state": row["event_state"],
                "tau_proxy_s": float(tau) if tau not in ("", None) else None,
            }
        )
    return frames


def _track_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for frame, boxes in _boxes_by_frame(rows).items():
        grouped[frame] = boxes
    return [{"frame": frame, "boxes": boxes} for frame, boxes in sorted(grouped.items())]


def _track_lengths(rows: list[dict]) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for row in rows:
        track_id = int(row["track_id"])
        lengths[track_id] = lengths.get(track_id, 0) + 1
    return lengths


def _episode_row(rows: list[dict], track_id: object, window: list[int]) -> dict:
    for row in rows:
        if str(row["track_id"]) == str(track_id) and [int(row["start_frame"]), int(row["end_frame"])] == window:
            return row
    return {}


def _frame_count(reference: Path, candidate: Path) -> int:
    counts = []
    for path in (reference, candidate):
        metadata = path / "metadata.json"
        if metadata.is_file():
            payload = json.loads(metadata.read_text())
            counts.append(int(payload["video"]["frames_processed"]))
    return max(counts) if counts else 0


def _run_record(path: Path) -> dict:
    metadata = json.loads((path / "metadata.json").read_text())
    return {
        "elapsed_seconds": metadata["timing"]["elapsed_seconds"],
        "frames_per_second": metadata["timing"]["frames_per_second"],
        "frames_processed": metadata["video"]["frames_processed"],
        "observation_count": metadata["counts"]["observations"],
        "episode_count": metadata["counts"]["episodes"],
        "device": metadata["configuration"]["detector"]["device"],
        "model_path": metadata["configuration"]["detector"]["model_path"],
        "profile": False,
    }


def _profile_record(path: Path) -> dict | None:
    summary_path = path / "profiling_summary.json"
    timings_path = path / "frame_timings.csv"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    warmup = None
    if timings_path.is_file():
        import csv

        with timings_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("warmup") == "true":
                    warmup = {
                        "frame_total_ms": float(row["frame_total_ms"]),
                        "inference_tracking_ms": float(row["inference_tracking_ms"]),
                    }
                    break
    stages = summary["stages"]
    return {
        "profile": True,
        "gpu_synchronization_enabled": summary["run"]["gpu_synchronization_enabled"],
        "pipeline_elapsed_seconds": summary["run"]["pipeline_elapsed_seconds"],
        "pipeline_frames_per_second": summary["run"]["pipeline_frames_per_second"],
        "measured_frames": summary["run"]["measured_frames"],
        "warmup": warmup,
        "frame_total_ms": stages["frame_total_ms"],
        "inference_tracking_ms": stages["inference_tracking_ms"],
        "note": (
            "Stage times exclude the first frame. Device cpu does not "
            "synchronize MPS. This run is not the throughput comparison."
        ),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile, method="linear"))


def render_report(summary: dict) -> str:
    historical = summary["coreml_versus_historical"]
    coreml = summary["runs"]["coreml"]
    lines = [
        "# Full-video Core ML validation",
        "",
        "Experimental only. The production CLI and default PyTorch/MPS backend were not changed.",
        "",
        "## Throughput",
        "",
        (
            f"Historical unprofiled PyTorch/MPS baseline: {REFERENCE_ELAPSED_SECONDS:.6f} s, "
            f"{REFERENCE_FPS:.6f} frames/s. This is not a paired measurement."
        ),
        (
            f"Core ML unprofiled run: {coreml['elapsed_seconds']} s, "
            f"{coreml['frames_per_second']} frames/s, {coreml['frames_processed']} frames, "
            f"model `{coreml['model_path']}` on {coreml['device']}."
        ),
        _speed_line(coreml["elapsed_seconds"], REFERENCE_ELAPSED_SECONDS, "the historical baseline"),
    ]
    fresh = summary["runs"].get("pytorch_mps")
    if fresh:
        lines.append(
            f"Fresh unprofiled PyTorch/MPS run: {fresh['elapsed_seconds']} s, "
            f"{fresh['frames_per_second']} frames/s."
        )
        lines.append(_speed_line(coreml["elapsed_seconds"], fresh["elapsed_seconds"], "the fresh PyTorch run"))
    profile = summary["runs"].get("coreml_profile")
    if profile:
        frame = profile["frame_total_ms"]
        warmup = profile.get("warmup") or {}
        lines.extend(
            [
                "",
                "## Frame latency",
                "",
                profile["note"],
                (
                    f"Profiled pipeline clock: {profile['pipeline_elapsed_seconds']} s. "
                    f"GPU synchronization enabled: {profile['gpu_synchronization_enabled']}."
                ),
                (
                    f"Steady frame total over {profile['measured_frames']} frames: "
                    f"mean {frame['mean_ms']} ms, median {frame['median_ms']} ms, "
                    f"p95 {frame['p95_ms']} ms, p99 {frame['p99_ms']} ms."
                ),
                (
                    f"Steady inference_tracking mean {profile['inference_tracking_ms']['mean_ms']} ms, "
                    f"p95 {profile['inference_tracking_ms']['p95_ms']} ms."
                ),
                (
                    f"First profiled frame: total {warmup.get('frame_total_ms', 'n/a')} ms, "
                    f"inference_tracking {warmup.get('inference_tracking_ms', 'n/a')} ms. "
                    "That frame contains Core ML compilation."
                ),
            ]
        )
    lines.extend(["", "## Historical output agreement", "", _comparison_block(historical)])
    if "coreml_versus_fresh_pytorch" in summary:
        lines.extend(["", "## Fresh PyTorch output agreement", "", _comparison_block(summary["coreml_versus_fresh_pytorch"])])
    lines.append("")
    return "\n".join(lines)


def _speed_line(coreml_seconds: float, other_seconds: float, label: str) -> str:
    delta = other_seconds - coreml_seconds
    ratio = other_seconds / coreml_seconds if coreml_seconds else 0
    return (
        f"Against {label}, Core ML is faster by {delta:.3f} s "
        f"({ratio:.3f}x end-to-end, {coreml_seconds:.3f} s versus {other_seconds:.3f} s)."
    )


def _comparison_block(compared: dict) -> str:
    counts = compared["counts"]
    detections = compared["detections"]
    tracking = compared["tracking"]
    categories = compared["categories"]
    episodes = compared["episodes"]
    lines = [
        (
            f"Observations: {counts['candidate_observations']} Core ML, "
            f"{counts['reference_observations']} reference. "
            f"{counts['frames_with_different_counts']} of {counts['frames']} frames "
            f"have a different count. Maximum absolute count change {counts['max_abs_count_delta']}."
        ),
        f"Classes: reference {compared['classes']['reference']}; Core ML {compared['classes']['candidate']}.",
        (
            f"Matched boxes {detections['matched']} of {detections['reference_boxes']} reference and "
            f"{detections['candidate_boxes']} Core ML. Median IoU {_num(detections['median_iou'])}, "
            f"p05 {_num(detections.get('p05_iou'))}, minimum {_num(detections['minimum_iou'])}. "
            f"Median center shift {_num(detections['median_center_px'])} px, "
            f"p95 {_num(detections.get('p95_center_px'))} px. "
            f"Median confidence gap {_num(detections['median_confidence_abs_diff'])}."
        ),
        (
            f"Trajectories: {tracking['mapped']} mapped of {tracking['reference_tracks']} reference and "
            f"{tracking['candidate_tracks']} Core ML. Same ID {tracking['same_id']}, "
            f"renumbered {tracking['renumbered']}, fragmented {tracking['fragmented_reference_tracks']}, "
            f"identity-switch frames {tracking['identity_switches']}, "
            f"unmatched reference {tracking['unmatched_reference']}, "
            f"unmatched candidate {tracking['unmatched_candidate']}. "
            f"Duration deltas of at least 5 frames: {tracking['duration']['pairs_differing_by_at_least_5']}. "
            f"Maximum absolute duration change {tracking['duration']['max_abs_frame_delta']} frames."
        ),
    ]
    for field in CATEGORY_FIELDS:
        item = categories[field]
        lines.append(f"{field}: {item['mismatched']} mismatches of {item['compared']} matched observations.")
    tau = categories["tau_proxy_s"]
    lines.append(
        f"tau_proxy_s on matched observations with both values: median absolute difference "
        f"{_num(tau['median_abs_diff'])}, p95 {_num(tau['p95_abs_diff'])}, "
        f"maximum {_num(tau['maximum_abs_diff'])}. Missing on one side: {tau['missing_on_one_side']}."
    )
    lines.append(
        f"Episodes: {episodes['candidate_episodes']} Core ML, {episodes['reference_episodes']} reference, "
        f"{episodes['mapped']} mapped, {episodes['same_bounds']} with the same frame bounds."
    )
    for pair in episodes["pairs"]:
        lines.append(_episode_line(pair))
    for row in episodes["unmatched_reference"]:
        lines.append(f"Unmatched reference episode: track {row['track_id']} frames {row['start_frame']}-{row['end_frame']}.")
    for row in episodes["unmatched_candidate"]:
        lines.append(f"Unmatched Core ML episode: track {row['track_id']} frames {row['start_frame']}-{row['end_frame']}.")
    return "\n".join(lines)


def _episode_line(pair: dict) -> str:
    fields = pair["fields"]
    changed = [
        f"{name} {item['reference']} -> {item['candidate']}"
        for name, item in fields.items()
        if not item["equal"]
    ]
    change_text = ", ".join(changed) if changed else "recorded fields match"
    return (
        f"Episode reference track {pair['reference_track_id']} "
        f"{pair['reference_window'][0]}-{pair['reference_window'][1]} "
        f"and Core ML track {pair['candidate_track_id']} "
        f"{pair['candidate_window'][0]}-{pair['candidate_window'][1]}: {change_text}."
    )


def _num(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _coreml():
    import importlib.util

    module = sys.modules.get("benchmark_coreml")
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location("benchmark_coreml", ROOT / "scripts" / "benchmark_coreml.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_coreml"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
