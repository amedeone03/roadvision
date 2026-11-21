"""Build a performance report from one profiled RoadVision run.

The script reads the profiling CSV and JSON. It does not rerun the
video and does not change detection, tracking, or event thresholds.
Stage times are synchronized wall-clock measurements. They are not
GPU kernel durations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadvision.profiling import (  # noqa: E402
    FRAME_TIMINGS_NAME,
    PROFILING_SUMMARY_NAME,
    STAGE_NAMES,
)


COMPONENTS = (
    ("Inference and tracking", ("inference_tracking_ms",)),
    ("Lane processing", ("lane_detection_ms", "lane_temporal_ms")),
    ("Association and motion", ("lane_association_ms", "motion_event_ms")),
    ("Annotation", ("annotation_ms",)),
    ("Video writing", ("video_write_ms",)),
    (
        "Remaining frame work",
        ("decode_ms", "csv_write_ms", "other_ms"),
    ),
)
EXACT_OBSERVATION_FIELDS = (
    "class_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "lane_status",
    "lane_reason",
    "baseline_status",
    "baseline_reason",
    "lane_valid",
    "lane_geometry_valid",
    "lane_temporal_reason",
    "approach_state",
    "evidence",
    "event_state",
)
NUMERIC_STAGES = (*STAGE_NAMES, "other_ms", "frame_total_ms")
DEFAULT_COMMAND = (
    "python -m roadvision run --input data/videos/test_drive.mp4 "
    "--output runs/profile_full --profile"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a profiled RoadVision run. "
            "This does not change the pipeline."
        )
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--prefix-profile", type=Path, default=None)
    parser.add_argument("--prefix-control", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--command", default=DEFAULT_COMMAND)
    args = parser.parse_args(argv)

    profile_dir = args.profile_dir
    report_path = args.report or (profile_dir / "performance_report.md")
    timings = read_timings(profile_dir / FRAME_TIMINGS_NAME)
    summary = json.loads(
        (profile_dir / PROFILING_SUMMARY_NAME).read_text(encoding="utf-8")
    )
    metadata_path = profile_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else None
    )
    reference = None
    baseline_metadata = None
    if args.reference_dir is not None:
        reference = compare_outputs(profile_dir, args.reference_dir)
        baseline_path = args.reference_dir / "metadata.json"
        if baseline_path.is_file():
            baseline_metadata = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
    analysis = analyze_timings(timings, summary)
    observations_path = profile_dir / "observations.csv"
    if observations_path.is_file():
        attach_lane_geometry(analysis, timings, observations_path)
    prefix = prefix_overhead(args.prefix_profile, args.prefix_control)
    report = render_report(
        analysis,
        summary,
        metadata,
        reference,
        prefix,
        baseline_metadata,
        command=args.command,
        profile_dir=profile_dir,
        reference_dir=args.reference_dir,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_share_svg(report_path.with_name("stage_share.svg"), analysis)
    print(f"Wrote {report_path}")
    if reference is not None and not reference["exact_agreement"]:
        print("Profiled outputs disagree with the reference.", file=sys.stderr)
        return 1
    return 0


def read_timings(path: Path) -> list[dict[str, object]]:
    """Load frame timings with numeric stages and boolean flags."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {
                "frame": int(raw["frame"]),
                "warmup": _is_true(raw["warmup"]),
                "tracked_objects": int(raw["tracked_objects"]),
                "lane_association_active": _is_true(
                    raw["lane_association_active"]
                ),
            }
            for name in NUMERIC_STAGES:
                row[name] = float(raw[name])
            rows.append(row)
    return rows


def analyze_timings(
    rows: list[dict[str, object]],
    summary: dict,
) -> dict:
    """Derive stage, component, and variability statistics.

    Warmup rows stay out of every aggregate. ``other_ms`` remains
    inside the frame total, so component shares use that total as
    the denominator.
    """

    measured = [row for row in rows if not row["warmup"]]
    warmup = [row for row in rows if row["warmup"]]
    if not measured:
        raise ValueError("The timing file has no measured frames.")

    stages = {
        name: _distribution([float(row[name]) for row in measured])
        for name in NUMERIC_STAGES
    }
    frame_total = stages["frame_total_ms"]["total_ms"]
    components = []
    for label, names in COMPONENTS:
        per_frame = [
            sum(float(row[name]) for name in names) for row in measured
        ]
        folded = _distribution(per_frame)
        components.append(
            {
                "label": label,
                "stages": names,
                "total_ms": folded["total_ms"],
                "mean_ms": folded["mean_ms"],
                "p95_ms": folded["p95_ms"],
                "p99_ms": folded["p99_ms"],
                "percent": 100.0 * folded["total_ms"] / frame_total,
            }
        )
    residual = _reconciliation_residual(measured)
    summary_delta = _summary_delta(stages, summary)
    return {
        "recorded_frames": len(rows),
        "measured_frames": len(measured),
        "warmup_frames": len(warmup),
        "warmup_frame_total_ms": (
            float(warmup[0]["frame_total_ms"]) if warmup else None
        ),
        "warmup_stages": (
            {
                name: float(warmup[0][name])
                for name in NUMERIC_STAGES
            }
            if warmup
            else None
        ),
        "stages": stages,
        "components": components,
        "reconciliation_max_abs_ms": residual,
        "summary_max_abs_total_delta_ms": summary_delta,
        "slowest": _slowest(measured),
        "objects": _object_association(measured),
        "video_write": _video_write(measured),
        "time_bins": _time_bins(measured, bins=10),
        "frame_index_correlation": _correlation(
            [float(row["frame"]) for row in measured],
            [float(row["frame_total_ms"]) for row in measured],
        ),
    }


def attach_lane_geometry(
    analysis: dict,
    timings: list[dict[str, object]],
    observations_path: Path,
) -> None:
    """Compare frame cost by the stored lane-geometry flag.

    The flag is copied onto every observation in a frame. Frames
    with no observation have no stored flag, so they stay in a
    separate count. This is a descriptive split, not a cause.
    """

    measured = {
        int(row["frame"]): row
        for row in timings
        if not row["warmup"]
    }
    flags: dict[int, str] = {}
    conflicts = 0
    with observations_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            frame = int(raw["frame"])
            flag = raw["lane_geometry_valid"]
            previous = flags.get(frame)
            if previous is None:
                flags[frame] = flag
            elif previous != flag:
                conflicts += 1
    groups = {"true": [], "false": [], "unobserved": []}
    for frame, row in measured.items():
        flag = flags.get(frame)
        if flag is None:
            groups["unobserved"].append(row)
        elif _is_true(flag):
            groups["true"].append(row)
        else:
            groups["false"].append(row)
    analysis["lane_geometry"] = {
        "conflicts": conflicts,
        "groups": {
            name: _group_cost(members)
            for name, members in groups.items()
        },
    }


def compare_outputs(profile_dir: Path, reference_dir: Path) -> dict:
    """Require the same keys, boxes, categories, and episode rows."""

    observations = _compare_csv(
        profile_dir / "observations.csv",
        reference_dir / "observations.csv",
        key=("frame", "track_id"),
        fields=EXACT_OBSERVATION_FIELDS,
    )
    events = _compare_csv(
        profile_dir / "events.csv",
        reference_dir / "events.csv",
        key=(
            "track_id",
            "start_frame",
            "end_frame",
            "confirmation_frame",
        ),
        fields=(
            "duration_frames",
            "duration_s",
            "positive_frames",
            "interruption_frames",
            "min_tau_proxy_s",
        ),
    )
    return {
        "observations": observations,
        "events": events,
        "exact_agreement": (
            observations["mismatched_rows"] == 0
            and observations["missing"] == 0
            and observations["extra"] == 0
            and events["mismatched_rows"] == 0
            and events["missing"] == 0
            and events["extra"] == 0
        ),
    }


def prefix_overhead(
    profile_dir: Path | None,
    control_dir: Path | None,
) -> dict | None:
    """Wall-clock difference for a short profiled and unprofiled pair."""

    if profile_dir is None or control_dir is None:
        return None
    profiled = json.loads(
        (profile_dir / "metadata.json").read_text(encoding="utf-8")
    )
    control = json.loads(
        (control_dir / "metadata.json").read_text(encoding="utf-8")
    )
    profiled_s = float(profiled["timing"]["elapsed_seconds"])
    control_s = float(control["timing"]["elapsed_seconds"])
    frames = int(profiled["video"]["frames_processed"])
    return {
        "frames": frames,
        "profiled_seconds": profiled_s,
        "control_seconds": control_s,
        "added_seconds": profiled_s - control_s,
        "added_ms_per_frame": 1000.0 * (profiled_s - control_s) / frames,
    }


def render_report(
    analysis: dict,
    summary: dict,
    metadata: dict | None,
    reference: dict | None,
    prefix: dict | None,
    baseline_metadata: dict | None,
    *,
    command: str,
    profile_dir: Path,
    reference_dir: Path | None,
) -> str:
    environment = summary["environment"]
    run = summary["run"]
    lines = [
        "# RoadVision full-video performance",
        "",
        "Synchronized wall-clock times from one profiled run. "
        "These are not GPU kernel durations. Device synchronization "
        "runs only while profiling is enabled, and it can slow the "
        "pipeline relative to an ordinary run.",
        "",
        "## Command and environment",
        "",
        "```text",
        command,
        "```",
        "",
        f"- Profile directory: `{profile_dir}`",
        f"- Python {environment['python']} ({environment['implementation']})",
        f"- Operating system: {environment['operating_system']}",
        f"- CPU architecture: {environment['cpu_architecture']}",
        f"- Torch {environment['torch']}, Ultralytics {environment['ultralytics']}",
        f"- Device: {run['device']}",
        f"- Model: {run['model_path']}",
        f"- Tracker: {run['tracker_path']}",
        f"- Confidence: {run['confidence']}",
        f"- Classes: {run['classes']}",
        (
            f"- Video: {run['video_path']} "
            f"({run['width']}x{run['height']}, {run['fps']} fps)"
        ),
        f"- Frames processed: {run['frames_processed']}",
        (
            "- GPU synchronization enabled: "
            f"{run['gpu_synchronization_enabled']} "
            f"({run['gpu_synchronization_calls']} calls)"
        ),
        _warmup_line(analysis),
        (
            "- Percentile sample: "
            f"{analysis['measured_frames']} measured frames; "
            f"sufficient = {run['percentile_sample_sufficient']}."
        ),
        "",
        _configuration_lines(metadata),
        "## Timing boundaries",
        "",
        "Stage timers do not overlap. `frame_total_ms` is the wall clock "
        "from the start of decoding through the observation flush. "
        "`other_ms` is the residual inside that interval. "
        "Model construction, reader and writer setup, episode "
        "finalization, and `events.csv` are outside the frame total "
        "and inside the pipeline clock. `metadata.json` and the "
        "profiling files are written after that clock stops.",
        "",
        (
            f"Largest absolute gap between `frame_total_ms` and the sum "
            f"of the exclusive stages plus `other_ms`: "
            f"{analysis['reconciliation_max_abs_ms']:.6f} ms."
        ),
        (
            f"Largest absolute gap between those measured totals and "
            f"`profiling_summary.json`: "
            f"{analysis['summary_max_abs_total_delta_ms']:.6f} ms."
        ),
        "",
        _outside_clock_line(summary, analysis),
        "",
        "## Behavioral agreement",
        "",
        _agreement_section(reference, reference_dir),
        "",
        "## Stage timings",
        "",
        "Totals, means, and percentiles use the measured frames only. "
        "The warm-up frame is not included.",
        "",
        _stage_table(analysis),
        "",
        "## Component share",
        "",
        "Percentages use measured `frame_total_ms` as the denominator. "
        "Inference and tracking includes the MPS or CUDA synchronization "
        "that profiling adds before that timer stops.",
        "",
        _component_table(analysis),
        "",
        "![Share of measured frame time](stage_share.svg)",
        "",
        "## Throughput and synchronization",
        "",
        _throughput_section(
            summary,
            analysis,
            metadata,
            baseline_metadata,
            prefix,
        ),
        "",
        "## Variability",
        "",
        "The splits below are descriptive. A difference between groups "
        "does not show that the grouping variable caused the extra time.",
        "",
        _slowest_section(analysis),
        "",
        _object_section(analysis),
        "",
        _lane_section(analysis),
        "",
        _video_section(analysis),
        "",
        _time_section(analysis),
        "",
        "## Bottleneck ranking",
        "",
        _ranking_section(analysis),
        "",
        "## Optimization opportunities",
        "",
        _opportunity_section(analysis),
        "",
    ]
    return "\n".join(lines)


def write_share_svg(path: Path, analysis: dict) -> None:
    """Write a horizontal bar chart of component share."""

    components = sorted(
        analysis["components"],
        key=lambda item: item["percent"],
        reverse=True,
    )
    row_height = 28
    width = 640
    height = 36 + row_height * len(components)
    bars = []
    for index, component in enumerate(components):
        y = 28 + index * row_height
        length = max(1.0, 360.0 * component["percent"] / 100.0)
        label = (
            f"{component['label']}  {component['percent']:.1f}%"
        )
        bars.append(
            f'<text x="8" y="{y - 6}" font-size="12" '
            f'fill="#1f2933">{_xml(label)}</text>'
            f'<rect x="8" y="{y}" width="{length:.1f}" height="12" '
            f'fill="#3d5a80"/>'
        )
    document = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Share of measured frame time">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        + "".join(bars)
        + "</svg>\n"
    )
    path.write_text(document, encoding="utf-8")


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "total_ms": float(array.sum()),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
        "p99_ms": float(np.percentile(array, 99, method="linear")),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _reconciliation_residual(rows: list[dict[str, object]]) -> float:
    worst = 0.0
    for row in rows:
        parts = sum(float(row[name]) for name in STAGE_NAMES)
        parts += float(row["other_ms"])
        worst = max(worst, abs(parts - float(row["frame_total_ms"])))
    return worst


def _summary_delta(stages: dict, summary: dict) -> float:
    worst = 0.0
    for name, stats in stages.items():
        exported = summary["stages"][name]["total_ms"]
        worst = max(worst, abs(stats["total_ms"] - float(exported)))
    return worst


def _slowest(rows: list[dict[str, object]]) -> dict:
    count = max(1, math.ceil(0.01 * len(rows)))
    ordered = sorted(
        rows,
        key=lambda row: float(row["frame_total_ms"]),
        reverse=True,
    )
    chosen = ordered[:count]
    rest = ordered[count:]
    return {
        "count": count,
        "frames": [int(row["frame"]) for row in chosen],
        "chosen": _group_cost(chosen),
        "rest": _group_cost(rest),
        "mean_tracked_objects": float(
            np.mean([int(row["tracked_objects"]) for row in chosen])
        ),
    }


def _object_association(rows: list[dict[str, object]]) -> dict:
    counts = [float(row["tracked_objects"]) for row in rows]
    totals = [float(row["frame_total_ms"]) for row in rows]
    inference = [float(row["inference_tracking_ms"]) for row in rows]
    bins: dict[str, list[dict[str, object]]] = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4+": [],
    }
    for row in rows:
        tracked = int(row["tracked_objects"])
        label = "4+" if tracked >= 4 else str(tracked)
        bins[label].append(row)
    return {
        "frame_total_correlation": _correlation(counts, totals),
        "inference_correlation": _correlation(counts, inference),
        "bins": {
            label: _group_cost(members) for label, members in bins.items()
        },
    }


def _video_write(rows: list[dict[str, object]]) -> dict:
    values = np.asarray(
        [float(row["video_write_ms"]) for row in rows],
        dtype=float,
    )
    threshold = float(np.percentile(values, 99, method="linear"))
    slow_frames = {
        int(row["frame"])
        for row in rows
        if float(row["video_write_ms"]) >= threshold
    }
    slowest_frames = set(_slowest(rows)["frames"])
    return {
        "median_ms": float(np.percentile(values, 50, method="linear")),
        "p99_ms": threshold,
        "max_ms": float(values.max()),
        "frames_at_or_above_p99": len(slow_frames),
        "overlap_with_slowest_frame_total": len(
            slow_frames & slowest_frames
        ),
    }


def _time_bins(
    rows: list[dict[str, object]],
    bins: int,
) -> list[dict[str, float | int]]:
    ordered = sorted(rows, key=lambda row: int(row["frame"]))
    chunks = np.array_split(np.asarray(ordered, dtype=object), bins)
    result = []
    for chunk in chunks:
        members = list(chunk)
        if not members:
            continue
        totals = [float(row["frame_total_ms"]) for row in members]
        result.append(
            {
                "start_frame": int(members[0]["frame"]),
                "end_frame": int(members[-1]["frame"]),
                "count": len(members),
                "mean_ms": float(np.mean(totals)),
                "p95_ms": float(np.percentile(totals, 95, method="linear")),
            }
        )
    return result


def _group_cost(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "count": 0,
            "mean_frame_total_ms": None,
            "mean_inference_tracking_ms": None,
            "mean_lane_detection_ms": None,
            "mean_video_write_ms": None,
        }
    return {
        "count": len(rows),
        "mean_frame_total_ms": float(
            np.mean([float(row["frame_total_ms"]) for row in rows])
        ),
        "mean_inference_tracking_ms": float(
            np.mean([float(row["inference_tracking_ms"]) for row in rows])
        ),
        "mean_lane_detection_ms": float(
            np.mean([float(row["lane_detection_ms"]) for row in rows])
        ),
        "mean_video_write_ms": float(
            np.mean([float(row["video_write_ms"]) for row in rows])
        ),
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _compare_csv(
    new_path: Path,
    reference_path: Path,
    *,
    key: tuple[str, ...],
    fields: tuple[str, ...],
) -> dict:
    new_rows = _read_csv(new_path)
    reference_rows = _read_csv(reference_path)
    new_index = _index_rows(new_rows, key)
    reference_index = _index_rows(reference_rows, key)
    missing = sorted(set(reference_index) - set(new_index))
    extra = sorted(set(new_index) - set(reference_index))
    mismatched = []
    for identity in sorted(set(reference_index) & set(new_index)):
        left = reference_index[identity]
        right = new_index[identity]
        if len(left) != len(right):
            mismatched.append((identity, "row_count", len(left), len(right)))
            continue
        for offset, (reference, candidate) in enumerate(zip(left, right)):
            for field in fields:
                if reference[field] != candidate[field]:
                    mismatched.append(
                        (identity, field, reference[field], candidate[field])
                    )
                    if len(mismatched) >= 20:
                        break
            if len(mismatched) >= 20:
                break
        if len(mismatched) >= 20:
            break
    return {
        "reference_rows": len(reference_rows),
        "new_rows": len(new_rows),
        "shared_keys": len(set(reference_index) & set(new_index)),
        "missing": len(missing),
        "extra": len(extra),
        "mismatched_rows": len(mismatched),
        "examples": mismatched[:8],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_rows(
    rows: list[dict[str, str]],
    key: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    indexed: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        identity = tuple(row[name] for name in key)
        indexed.setdefault(identity, []).append(row)
    return indexed


def _warmup_line(analysis: dict) -> str:
    total = _ms(analysis["warmup_frame_total_ms"])
    stages = analysis.get("warmup_stages")
    detail = ""
    if stages is not None:
        detail = (
            f" Inference and tracking accounts for "
            f"{stages['inference_tracking_ms']:.3f} ms of that frame."
        )
    return (
        "- Warm-up: the first completed frame is stored and excluded "
        f"from the statistics below ({analysis['warmup_frames']} frame, "
        f"{total} ms).{detail} The excluded frame is not part of the "
        "exported stage totals."
    )


def _configuration_lines(metadata: dict | None) -> str:
    if metadata is None:
        return "Pipeline metadata was not present beside the timings.\n"
    video = metadata["video"]
    timing = metadata["timing"]
    counts = metadata["counts"]
    return "\n".join(
        [
            "- Effective configuration from `metadata.json`:",
            f"  - device `{metadata['model']['device']}`, model `{metadata['model']['path']}`",
            (
                f"  - {video['frames_processed']} frames at "
                f"{video['width']}x{video['height']}, "
                f"container rate {video['fps']}"
            ),
            (
                f"  - observations {counts['observations']}, "
                f"episodes {counts['episodes']}, "
                f"accepted lane frames {counts['accepted_lane_frames']}"
            ),
            (
                f"  - pipeline clock {timing['elapsed_seconds']} s, "
                f"{timing['frames_per_second']} frames/s"
            ),
            "",
        ]
    )


def _outside_clock_line(summary: dict, analysis: dict) -> str:
    elapsed = summary["run"]["pipeline_elapsed_seconds"]
    if not isinstance(elapsed, (int, float)):
        return "Pipeline elapsed time was not recorded."
    recorded_ms = analysis["stages"]["frame_total_ms"]["total_ms"]
    if analysis["warmup_frame_total_ms"] is not None:
        recorded_ms += analysis["warmup_frame_total_ms"]
    outside = float(elapsed) - recorded_ms / 1000.0
    return (
        f"Pipeline clock: {float(elapsed):.3f} s. "
        f"Recorded frame time, including warm-up: {recorded_ms / 1000.0:.3f} s. "
        f"Time outside the frame timers: {outside:.3f} s. "
        "That remainder is model construction, reader and writer setup, "
        "episode finalization, and `events.csv`."
    )


def _agreement_section(reference: dict | None, reference_dir: Path | None) -> str:
    if reference is None:
        return "No reference directory was supplied."
    observations = reference["observations"]
    events = reference["events"]
    status = (
        "Exact agreement on observation keys, rounded boxes, lane fields, "
        "approach state, evidence, event state, and episode rows."
        if reference["exact_agreement"]
        else "The profiled outputs do not match the reference."
    )
    lines = [
        f"Reference: `{reference_dir}`.",
        "",
        status,
        "",
        (
            f"- Observations: {observations['new_rows']} profiled rows, "
            f"{observations['reference_rows']} reference rows, "
            f"{observations['shared_keys']} shared keys, "
            f"{observations['missing']} missing, "
            f"{observations['extra']} extra, "
            f"{observations['mismatched_rows']} field mismatches."
        ),
        (
            f"- Events: {events['new_rows']} profiled rows, "
            f"{events['reference_rows']} reference rows, "
            f"{events['shared_keys']} shared keys, "
            f"{events['missing']} missing, "
            f"{events['extra']} extra, "
            f"{events['mismatched_rows']} field mismatches."
        ),
    ]
    examples = observations["examples"] or events["examples"]
    if examples:
        lines.extend(["", "First mismatches:", ""])
        for identity, field, left, right in examples:
            lines.append(
                f"- `{identity}` `{field}`: reference `{left}`, "
                f"profiled `{right}`"
            )
    return "\n".join(lines)


def _stage_table(analysis: dict) -> str:
    header = (
        "| Stage | Total ms | Mean | Median | p95 | p99 | Min | Max | Share |"
    )
    frame_total = analysis["stages"]["frame_total_ms"]["total_ms"]
    body = [header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in (*STAGE_NAMES, "other_ms", "frame_total_ms"):
        stats = analysis["stages"][name]
        share = 100.0 * stats["total_ms"] / frame_total
        body.append(
            "| {name} | {total:.1f} | {mean:.3f} | {median:.3f} | "
            "{p95:.3f} | {p99:.3f} | {minimum:.3f} | {maximum:.3f} | "
            "{share:.1f}% |".format(
                name=f"`{name}`",
                total=stats["total_ms"],
                mean=stats["mean_ms"],
                median=stats["median_ms"],
                p95=stats["p95_ms"],
                p99=stats["p99_ms"],
                minimum=stats["min_ms"],
                maximum=stats["max_ms"],
                share=share,
            )
        )
    return "\n".join(body)


def _component_table(analysis: dict) -> str:
    body = [
        "| Component | Total s | Mean ms | Share |",
        "|---|---:|---:|---:|",
    ]
    for component in analysis["components"]:
        body.append(
            f"| {component['label']} | {component['total_ms'] / 1000.0:.3f} | "
            f"{component['mean_ms']:.3f} | {component['percent']:.1f}% |"
        )
    return "\n".join(body)


def _throughput_section(
    summary: dict,
    analysis: dict,
    metadata: dict | None,
    baseline_metadata: dict | None,
    prefix: dict | None,
) -> str:
    run = summary["run"]
    mean_ms = analysis["stages"]["frame_total_ms"]["mean_ms"]
    steady = 1000.0 / mean_ms
    lines = [
        (
            f"Profiled pipeline throughput: "
            f"{run['pipeline_frames_per_second']} frames/s "
            f"over {run['pipeline_elapsed_seconds']} s. "
            "This clock includes startup and finalization, and it "
            "includes GPU synchronization."
        ),
        (
            f"Steady measured frame time: mean {mean_ms:.3f} ms, "
            f"which is {steady:.3f} frames/s if every measured frame "
            "kept that mean. Warm-up, startup, finalization, and the "
            "profiling export are not in this rate."
        ),
        (
            "The published unprofiled v1.0 measurement on this video "
            "was 34.936 frames/s."
        ),
    ]
    if metadata is not None and "timing" in metadata:
        profiled = float(metadata["timing"]["elapsed_seconds"])
        lines.append(
            f"This profiled run's own metadata clock is {profiled:.3f} s "
            f"({metadata['timing']['frames_per_second']} frames/s)."
        )
    if baseline_metadata is not None and "timing" in baseline_metadata:
        baseline_fps = float(baseline_metadata["timing"]["frames_per_second"])
        baseline_s = float(baseline_metadata["timing"]["elapsed_seconds"])
        lines.append(
            "The stored unprofiled `runs/full_regression` metadata records "
            f"{baseline_fps:.6f} frames/s over {baseline_s:.3f} s. "
            "That stored rate rounds to the published 34.936 frames/s."
        )
        profiled_fps = run["pipeline_frames_per_second"]
        if isinstance(profiled_fps, (int, float)) and profiled_fps > 0:
            lines.append(
                f"The profiled pipeline rate is {float(profiled_fps):.3f} "
                "frames/s. Baseline minus profiled is "
                f"{baseline_fps - float(profiled_fps):.3f} frames/s. "
                "Part of that gap is synchronization and timer overhead. "
                "The two full runs were not started together, so the "
                "difference is not a controlled overhead measurement."
            )
    if prefix is not None:
        lines.append(
            "The earlier same-session "
            f"{prefix['frames']}-frame pair is a tighter overhead check: "
            f"profiled {prefix['profiled_seconds']:.3f} s, unprofiled "
            f"{prefix['control_seconds']:.3f} s, difference "
            f"{prefix['added_seconds']:.3f} s "
            f"({prefix['added_ms_per_frame']:.2f} ms/frame). "
            "That pair still includes startup in both clocks."
        )
    lines.append(
        "`inference_tracking_ms` waits for MPS or CUDA before the timer "
        "stops. An ordinary run does not add that wait. The stage is "
        "therefore an upper bound on the unsynchronized inference wall "
        "time, and it is still not a kernel latency."
    )
    return "\n\n".join(lines)


def _slowest_section(analysis: dict) -> str:
    slow = analysis["slowest"]
    chosen = slow["chosen"]
    rest = slow["rest"]
    return "\n".join(
        [
            (
                f"Slowest 1% is {slow['count']} measured frames "
                f"(ceiling of 1% of {analysis['measured_frames']})."
            ),
            (
                f"- Slow set mean frame time {_num(chosen['mean_frame_total_ms'])} ms, "
                f"inference {_num(chosen['mean_inference_tracking_ms'])} ms, "
                f"lane detection {_num(chosen['mean_lane_detection_ms'])} ms, "
                f"video write {_num(chosen['mean_video_write_ms'])} ms, "
                f"mean tracked objects {slow['mean_tracked_objects']:.2f}."
            ),
            (
                f"- Other frames mean frame time {_num(rest['mean_frame_total_ms'])} ms, "
                f"inference {_num(rest['mean_inference_tracking_ms'])} ms, "
                f"lane detection {_num(rest['mean_lane_detection_ms'])} ms, "
                f"video write {_num(rest['mean_video_write_ms'])} ms."
            ),
            f"- Frame numbers: {', '.join(str(frame) for frame in slow['frames'])}.",
        ]
    )


def _object_section(analysis: dict) -> str:
    objects = analysis["objects"]
    lines = [
        (
            "Pearson correlation between tracked-object count and "
            f"frame time: {_corr(objects['frame_total_correlation'])}. "
            "The same count against inference and tracking: "
            f"{_corr(objects['inference_correlation'])}."
        ),
        "",
        "| Tracked objects | Frames | Mean frame ms | Mean inference ms |",
        "|---|---:|---:|---:|",
    ]
    for label in ("0", "1", "2", "3", "4+"):
        group = objects["bins"][label]
        lines.append(
            f"| {label} | {group['count']} | "
            f"{_num(group['mean_frame_total_ms'])} | "
            f"{_num(group['mean_inference_tracking_ms'])} |"
        )
    return "\n".join(lines)


def _lane_section(analysis: dict) -> str:
    lane = analysis.get("lane_geometry")
    if lane is None:
        return (
            "Lane-geometry groups were not attached. "
            "Pass the observations file through `attach_lane_geometry`."
        )
    lines = [
        (
            "Groups use `lane_geometry_valid` from `observations.csv`. "
            f"Conflicting flags inside one frame: {lane['conflicts']}."
        ),
        "",
        "| Lane geometry | Frames | Mean frame ms | Mean lane detection ms |",
        "|---|---:|---:|---:|",
    ]
    for label in ("true", "false", "unobserved"):
        group = lane["groups"][label]
        lines.append(
            f"| {label} | {group['count']} | "
            f"{_num(group['mean_frame_total_ms'])} | "
            f"{_num(group['mean_lane_detection_ms'])} |"
        )
    return "\n".join(lines)


def _video_section(analysis: dict) -> str:
    video = analysis["video_write"]
    return (
        f"Video-write median {_num(video['median_ms'])} ms, "
        f"p99 {_num(video['p99_ms'])} ms, max {_num(video['max_ms'])} ms. "
        f"{video['frames_at_or_above_p99']} frames are at or above p99, "
        f"and {video['overlap_with_slowest_frame_total']} of those are "
        "also in the slowest 1% of frame totals."
    )


def _time_section(analysis: dict) -> str:
    lines = [
        (
            "Correlation between frame number and measured frame time: "
            f"{_corr(analysis['frame_index_correlation'])}."
        ),
        "",
        "| Frames | Count | Mean ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for chunk in analysis["time_bins"]:
        lines.append(
            f"| {chunk['start_frame']}–{chunk['end_frame']} | "
            f"{chunk['count']} | {chunk['mean_ms']:.3f} | "
            f"{chunk['p95_ms']:.3f} |"
        )
    return "\n".join(lines)


def _ranking_section(analysis: dict) -> str:
    ranked = sorted(
        analysis["components"],
        key=lambda item: item["total_ms"],
        reverse=True,
    )
    lines = []
    for index, component in enumerate(ranked, start=1):
        lines.append(
            f"{index}. {component['label']}: {component['percent']:.1f}% "
            f"of measured frame time ({component['total_ms'] / 1000.0:.3f} s)."
        )
    return "\n".join(lines)


def _opportunity_section(analysis: dict) -> str:
    ranked = sorted(
        analysis["components"],
        key=lambda item: item["total_ms"],
        reverse=True,
    )
    notes = {
        "Inference and tracking": (
            "This is one Ultralytics `track()` call. YOLO and ByteTrack "
            "are not separate measurements. The timer includes the "
            "profiling synchronization, so a later change should be "
            "judged on an unprofiled run."
        ),
        "Lane processing": (
            "Almost all of this time is classical lane detection. "
            "Temporal filtering is a small part of the same component."
        ),
        "Video writing": (
            "The annotated frame is encoded on the same thread as "
            "perception. A high p99 relative to the median is a latency "
            "spike, not a reason to treat the median as the whole cost."
        ),
        "Annotation": (
            "Drawing is smaller than detection and encoding, so it is "
            "not the first place to change."
        ),
        "Association and motion": (
            "Vehicle-to-lane association and event updates are a small "
            "share of the frame. They are not the current performance "
            "limit."
        ),
        "Remaining frame work": (
            "Decoding, CSV flushing, and the untimed residual are the "
            "rest of the frame. They are visible, and they are smaller "
            "than the three stages above them in this ranking."
        ),
    }
    lines = []
    for index, component in enumerate(ranked[:3], start=1):
        lines.append(
            f"{index}. **{component['label']}** — {component['percent']:.1f}% "
            f"of measured frame time, mean {component['mean_ms']:.3f} ms, "
            f"p95 {component['p95_ms']:.3f} ms, p99 {component['p99_ms']:.3f} ms. "
            f"{notes[component['label']]}"
        )
    return "\n".join(lines)


def _is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _corr(value: float | None) -> str:
    if value is None:
        return "not computed"
    return f"{value:.3f}"


def _xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
