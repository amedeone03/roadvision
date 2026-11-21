"""Compare one live RoadVision run with the Phase 6 CSV baseline.

The script reports differences. It does not change thresholds or
rewrite either result. A shared ``(frame, track_id)`` key means the
two files use the same tracker identity at that frame. It does not
by itself prove that the identity is the same physical object.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadvision.config import (  # noqa: E402
    EVENT_STATES,
    LANE_STATUSES,
    MOTION_STATES,
    OBSERVATION_COLUMNS,
    PHASE6_DURATION_DIVISOR,
)
from roadvision.pipeline import EVENT_COLUMNS  # noqa: E402


EVIDENCE_STATES = (
    "MOTION_UNAVAILABLE",
    "NO_APPROACH_EVIDENCE",
    "EGO_LANE_APPROACH",
    "OTHER_LANE_APPROACH",
    "LANE_UNCERTAIN_APPROACH",
)
BOX_COLUMNS = ("x1", "y1", "x2", "y2")
DEFAULT_OUTPUT = Path("runs/full_regression")
DEFAULT_REFERENCE_OBSERVATIONS = Path(
    "runs/phase6/final/phase6_results.csv"
)
DEFAULT_REFERENCE_EVENTS = Path(
    "runs/phase6/stabilized_approach_events.csv"
)


@dataclass(frozen=True, slots=True)
class Episode:
    track_id: int
    start_frame: int
    end_frame: int
    confirmation_frame: int
    duration_frames: int
    positive_frames: int
    interruption_frames: int
    duration_s: float | None
    min_tau_proxy_s: float | None

    def bounds_equal(self, other: Episode) -> bool:
        return (
            self.start_frame == other.start_frame
            and self.confirmation_frame == other.confirmation_frame
            and self.end_frame == other.end_frame
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a live RoadVision run with the Phase 6 baseline. "
            "This does not retune either result."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-observations",
        type=Path,
        default=DEFAULT_REFERENCE_OBSERVATIONS,
    )
    parser.add_argument(
        "--reference-events",
        type=Path,
        default=DEFAULT_REFERENCE_EVENTS,
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--wall-clock-seconds", type=float, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    report_path = args.report or (output_dir / "comparison_report.md")
    required = {
        "live observations": output_dir / "observations.csv",
        "live events": output_dir / "events.csv",
        "live metadata": output_dir / "metadata.json",
        "reference observations": args.reference_observations,
        "reference events": args.reference_events,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.is_file()
    ]
    if missing:
        print("Missing comparison input:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1

    metadata = json.loads(
        required["live metadata"].read_text(encoding="utf-8")
    )
    new_rows = read_observation_rows(required["live observations"])
    reference_rows = read_observation_rows(
        required["reference observations"],
        approach_column="approach_t015",
    )
    new_episodes = read_episodes(required["live events"])
    reference_episodes = read_episodes(required["reference events"])
    comparison = compare_runs(
        reference_rows,
        new_rows,
        reference_episodes,
        new_episodes,
    )
    source_video = Path(metadata["video"]["path"])
    if not source_video.is_absolute():
        source_video = ROOT / source_video
    validation = validate_run(
        output_dir,
        metadata,
        new_rows,
        new_episodes,
        source_video=source_video,
    )
    report = render_report(
        comparison,
        validation,
        metadata,
        wall_clock_seconds=args.wall_clock_seconds,
        reference_observation_path=args.reference_observations,
        reference_event_path=args.reference_events,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    return 1 if validation["problems"] else 0


def read_observation_rows(
    path: Path,
    approach_column: str = "approach_state",
) -> list[dict]:
    frame = pd.read_csv(path)
    if approach_column not in frame.columns:
        raise ValueError(f"{path} has no {approach_column} column")
    rows = []
    for record in frame.to_dict(orient="records"):
        row = dict(record)
        row["approach_state"] = row[approach_column]
        rows.append(row)
    return rows


def read_episodes(path: Path) -> list[Episode]:
    frame = pd.read_csv(path)
    episodes = []
    for record in frame.to_dict(orient="records"):
        episodes.append(
            Episode(
                track_id=int(record["track_id"]),
                start_frame=int(record["start_frame"]),
                end_frame=int(record["end_frame"]),
                confirmation_frame=int(record["confirmation_frame"]),
                duration_frames=int(record["duration_frames"]),
                positive_frames=int(record["positive_frames"]),
                interruption_frames=int(record["interruption_frames"]),
                duration_s=_finite(record.get("duration_s")),
                min_tau_proxy_s=_finite(record.get("min_tau_proxy_s")),
            )
        )
    return episodes


def compare_runs(
    reference_rows: list[dict],
    new_rows: list[dict],
    reference_episodes: list[Episode],
    new_episodes: list[Episode],
) -> dict:
    reference_index, reference_duplicates = index_rows(reference_rows)
    new_index, new_duplicates = index_rows(new_rows)
    reference_keys = set(reference_index)
    new_keys = set(new_index)
    shared = reference_keys & new_keys
    episodes = compare_episodes(
        reference_episodes,
        new_episodes,
        reference_index,
        new_index,
    )
    episodes["reference_episodes_source"] = reference_episodes
    episodes["new_episodes_source"] = new_episodes
    return {
        "duplicates": {
            "reference": reference_duplicates,
            "new": new_duplicates,
        },
        "tracking": compare_tracking(
            reference_index,
            new_index,
            reference_keys,
            new_keys,
            shared,
        ),
        "lanes": compare_labels(
            reference_index,
            new_index,
            shared,
            "lane_status",
            LANE_STATUSES,
        ),
        "motion": compare_motion(reference_index, new_index, shared),
        "event_states": compare_labels(
            reference_index,
            new_index,
            shared,
            "event_state",
            EVENT_STATES,
        ),
        "evidence": compare_labels(
            reference_index,
            new_index,
            shared,
            "evidence",
            EVIDENCE_STATES,
        ),
        "episodes": episodes,
        "distributions": {
            "reference": distributions(reference_rows),
            "new": distributions(new_rows),
        },
    }


def compare_tracking(
    reference_index,
    new_index,
    reference_keys,
    new_keys,
    shared,
):
    corner_diffs = []
    coordinate_diffs = {column: [] for column in BOX_COLUMNS}
    ious = []
    for key in shared:
        reference = reference_index[key]
        new = new_index[key]
        corners = []
        for column in BOX_COLUMNS:
            difference = abs(
                float(reference[column]) - float(new[column])
            )
            coordinate_diffs[column].append(difference)
            corners.append(difference)
        corner_diffs.append(max(corners))
        ious.append(
            box_iou(
                [float(reference[column]) for column in BOX_COLUMNS],
                [float(new[column]) for column in BOX_COLUMNS],
            )
        )
    missing = sorted(reference_keys - new_keys)
    added = sorted(new_keys - reference_keys)
    return {
        "reference_observations": len(reference_keys),
        "new_observations": len(new_keys),
        "reference_tracks": _track_ids(reference_keys),
        "new_tracks": _track_ids(new_keys),
        "reference_frames": _frame_count(reference_keys),
        "new_frames": _frame_count(new_keys),
        "shared_keys": len(shared),
        "missing_keys": len(missing),
        "added_keys": len(added),
        "missing_examples": missing[:10],
        "added_examples": added[:10],
        "missing_by_track": _count_tracks(missing),
        "added_by_track": _count_tracks(added),
        "max_corner": summarize(corner_diffs),
        "coordinates": {
            column: summarize(values)
            for column, values in coordinate_diffs.items()
        },
        "iou": summarize(ious),
    }


def compare_labels(reference_index, new_index, shared, field, labels):
    matrix = {
        reference: {new: 0 for new in labels}
        for reference in labels
    }
    unexpected = 0
    matches = 0
    definite_changes = Counter()
    for key in shared:
        reference = str(reference_index[key][field])
        new = str(new_index[key][field])
        if reference == new:
            matches += 1
        if reference in matrix and new in matrix[reference]:
            matrix[reference][new] += 1
        else:
            unexpected += 1
        if field == "lane_status":
            definite_changes[_definiteness_change(reference, new)] += 1
    agreement = matches / len(shared) if shared else None
    return {
        "compared": len(shared),
        "matches": matches,
        "agreement": agreement,
        "matrix": matrix,
        "unexpected": unexpected,
        "definiteness": dict(definite_changes),
    }


def compare_motion(reference_index, new_index, shared):
    both_finite = []
    both_missing = 0
    reference_missing = 0
    new_missing = 0
    threshold_crossings = 0
    for key in shared:
        reference_growth = _finite(
            reference_index[key]["growth_robust_5"]
        )
        new_growth = _finite(new_index[key]["growth_robust_5"])
        if reference_growth is None and new_growth is None:
            both_missing += 1
        elif reference_growth is None:
            reference_missing += 1
        elif new_growth is None:
            new_missing += 1
        else:
            both_finite.append(abs(reference_growth - new_growth))
            if _growth_side(reference_growth) != _growth_side(new_growth):
                threshold_crossings += 1
    return {
        "both_finite": len(both_finite),
        "both_missing": both_missing,
        "reference_missing_only": reference_missing,
        "new_missing_only": new_missing,
        "growth": summarize(both_finite),
        "threshold_crossings": threshold_crossings,
        "approach": compare_labels(
            reference_index,
            new_index,
            shared,
            "approach_state",
            MOTION_STATES,
        ),
    }


def compare_episodes(
    reference_episodes,
    new_episodes,
    reference_index,
    new_index,
):
    grouped_reference = _group_tracks(reference_episodes)
    grouped_new = _group_tracks(new_episodes)
    exact = []
    shifted = []
    split = []
    merged = []
    ambiguous = []
    missing = []
    added = []
    tracks = set(grouped_reference) | set(grouped_new)
    for track_id in sorted(tracks):
        components = _components(
            grouped_reference.get(track_id, []),
            grouped_new.get(track_id, []),
        )
        for reference_group, new_group in components:
            pair = _boundary_record(reference_group, new_group)
            if reference_group and not new_group:
                missing.extend(reference_group)
            elif new_group and not reference_group:
                added.extend(new_group)
            elif len(reference_group) == 1 and len(new_group) == 1:
                if reference_group[0].bounds_equal(new_group[0]):
                    exact.append(pair)
                else:
                    pair["attribution"] = _attribute_shift(
                        reference_group[0],
                        new_group[0],
                        reference_index,
                        new_index,
                    )
                    shifted.append(pair)
            elif len(reference_group) == 1 and len(new_group) > 1:
                split.append(pair)
            elif len(reference_group) > 1 and len(new_group) == 1:
                merged.append(pair)
            else:
                ambiguous.append(pair)

    identity_candidates, unpaired_missing, unpaired_added = (
        _identity_candidates(missing, added)
    )
    one_to_one = exact + shifted
    tau_diffs = []
    for pair in exact:
        reference_tau = pair["reference"][0].min_tau_proxy_s
        new_tau = pair["new"][0].min_tau_proxy_s
        if reference_tau is not None and new_tau is not None:
            tau_diffs.append(abs(reference_tau - new_tau))
    return {
        "reference_count": len(reference_episodes),
        "new_count": len(new_episodes),
        "exact": exact,
        "shifted": shifted,
        "split": split,
        "merged": merged,
        "ambiguous": ambiguous,
        "missing": unpaired_missing,
        "added": unpaired_added,
        "identity_candidates": identity_candidates,
        "boundary_matches": {
            "pairs": len(one_to_one),
            "start": sum(item["start_equal"] for item in one_to_one),
            "confirmation": sum(
                item["confirmation_equal"] for item in one_to_one
            ),
            "end": sum(item["end_equal"] for item in one_to_one),
            "all_three": len(exact),
        },
        "exact_min_tau": summarize(tau_diffs),
    }


def validate_run(
    output_dir: Path,
    metadata: dict,
    rows: list[dict],
    episodes: list[Episode],
    source_video: Path | None = None,
) -> dict:
    problems = []
    observations_path = (
        output_dir / metadata["outputs"]["observations_csv"]
    )
    events_path = output_dir / metadata["outputs"]["events_csv"]
    if _csv_header(observations_path) != list(OBSERVATION_COLUMNS):
        problems.append(
            "observations.csv schema does not match the pipeline"
        )
    if _csv_header(events_path) != list(EVENT_COLUMNS):
        problems.append("events.csv schema does not match the pipeline")

    indexed, duplicates = index_rows(rows)
    if duplicates:
        problems.append(
            f"observations.csv has {len(duplicates)} duplicate keys"
        )
    if len(rows) != metadata["counts"]["observations"]:
        problems.append(
            "metadata observation count disagrees with the CSV"
        )
    if len(episodes) != metadata["counts"]["episodes"]:
        problems.append(
            "metadata episode count disagrees with events.csv"
        )

    fps = float(metadata["video"]["fps"])
    frames_processed = int(metadata["video"]["frames_processed"])
    for row in rows:
        frame = int(row["frame"])
        if frame < 1 or frame > frames_processed:
            problems.append(
                f"observation frame {frame} is outside the run"
            )
            break
        expected_time = (frame - 1) / fps
        actual_time = _finite(row["time_seconds"])
        if actual_time is None or abs(actual_time - expected_time) > 1e-9:
            problems.append(
                f"time_seconds disagrees with measured FPS at frame {frame}"
            )
            break

    for episode in episodes:
        problems.extend(
            _episode_problems(episode, fps, frames_processed, indexed)
        )

    video = _validate_video(
        output_dir / metadata["outputs"]["annotated_video"],
        metadata,
        source_video,
    )
    problems.extend(video["problems"])
    return {"problems": problems, "video": video}


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def box_iou(reference: list[float], new: list[float]) -> float:
    intersection_x1 = max(reference[0], new[0])
    intersection_y1 = max(reference[1], new[1])
    intersection_x2 = min(reference[2], new[2])
    intersection_y2 = min(reference[3], new[3])
    intersection = (
        max(0.0, intersection_x2 - intersection_x1)
        * max(0.0, intersection_y2 - intersection_y1)
    )
    reference_area = max(0.0, reference[2] - reference[0]) * max(
        0.0,
        reference[3] - reference[1],
    )
    new_area = max(0.0, new[2] - new[0]) * max(0.0, new[3] - new[1])
    union = reference_area + new_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def overlap_frames(reference: Episode, new: Episode) -> int:
    start = max(reference.start_frame, new.start_frame)
    end = min(reference.end_frame, new.end_frame)
    return max(0, end - start + 1)


def index_rows(rows: list[dict]) -> tuple[dict, list]:
    indexed = {}
    duplicates = []
    for row in rows:
        key = (int(row["frame"]), int(row["track_id"]))
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = row
    return indexed, duplicates


def distributions(rows: list[dict]) -> dict:
    keys = [(int(row["frame"]), int(row["track_id"])) for row in rows]
    return {
        "tracks": sorted({track for _, track in keys}),
        "frames_with_tracks": len({frame for frame, _ in keys}),
        "lane_status": dict(Counter(str(row["lane_status"]) for row in rows)),
        "approach_state": dict(
            Counter(str(row["approach_state"]) for row in rows)
        ),
        "evidence": dict(Counter(str(row.get("evidence")) for row in rows)),
        "event_state": dict(
            Counter(str(row["event_state"]) for row in rows)
        ),
    }


def render_report(
    comparison: dict,
    validation: dict,
    metadata: dict,
    wall_clock_seconds: float | None,
    reference_observation_path: Path,
    reference_event_path: Path,
) -> str:
    video = metadata["video"]
    timing = metadata["timing"]
    counts = metadata["counts"]
    model = metadata["model"]
    new_dist = comparison["distributions"]["new"]
    episodes = comparison["episodes"]
    notes = findings(comparison, metadata)
    video_check = validation["video"]
    wall_clock = (
        wall_clock_seconds
        if wall_clock_seconds is not None
        else "not recorded"
    )
    lines = [
        "# Full-video regression",
        "",
        "This comparison does not change thresholds. Shared "
        "`(frame, track_id)` keys are tracker identities, not "
        "confirmed physical objects. Completed episodes are "
        "algorithmic image-space evidence, not collision warnings.",
        "",
        "## Execution",
        "",
        f"- Input: `{video['path']}`",
        f"- Resolution: {video['width']}x{video['height']}",
        f"- Measured FPS: {video['fps']}",
        f"- Container frames: {video['container_frame_count']}",
        f"- Frames processed: {video['frames_processed']}",
        f"- Model: `{model['path']}`",
        f"- Device: {model['device']}",
        f"- Confidence: {model['confidence']}",
        f"- Classes: {model['classes']}",
        (
            "- Unrounded detector boxes used for growth: "
            f"{metadata['formatting']['motion_uses_unrounded_detector_boxes']}"
        ),
        f"- Observations: {counts['observations']}",
        f"- Unique track IDs: {len(new_dist['tracks'])}",
        f"- Track IDs: {', '.join(map(str, new_dist['tracks']))}",
        f"- Frames containing tracks: {new_dist['frames_with_tracks']}",
        f"- Accepted lane frames: {counts['accepted_lane_frames']}",
        f"- Pipeline runtime: {timing['elapsed_seconds']} s",
        (
            "- Processing throughput: "
            f"{timing['frames_per_second']} frames/s"
        ),
        f"- Hardware: {_hardware()}",
        "",
        "### Live label counts",
        "",
        _count_line("Lane", new_dist["lane_status"]),
        _count_line("Motion", new_dist["approach_state"]),
        _count_line("Evidence", new_dist["evidence"]),
        _count_line("Event state", new_dist["event_state"]),
        "",
        "### Completed evidence episodes",
        "",
        _episode_table(episodes["new_episodes_source"]),
        "",
        "## Artifact validation",
        "",
    ]
    if validation["problems"]:
        lines.extend(f"- {problem}" for problem in validation["problems"])
    else:
        lines.append(
            "- Annotated video, CSV schemas, keys, metadata counts, "
            "and event bounds agree."
        )
    lines.extend([
        f"- Decoded frames: {video_check.get('decoded_frames')}",
        (
            "- Decoded size: "
            f"{video_check.get('width')}x{video_check.get('height')}"
        ),
        f"- Output file size: {video_check.get('file_bytes')} bytes",
        (
            "- Sampled mean absolute difference from the source: "
            f"{video_check.get('sampled_mean_abs_diff')}"
        ),
        "",
        "## Detection and tracking",
        "",
        f"Reference observations: `{reference_observation_path}`",
        "",
    ])
    tracking = comparison["tracking"]
    lines.extend([
        f"- Reference observations: {tracking['reference_observations']}",
        f"- Live observations: {tracking['new_observations']}",
        f"- Reference tracks: {len(tracking['reference_tracks'])}",
        f"- Live tracks: {len(tracking['new_tracks'])}",
        f"- Reference frames with tracks: {tracking['reference_frames']}",
        f"- Live frames with tracks: {tracking['new_frames']}",
        f"- Shared keys: {tracking['shared_keys']}",
        f"- Keys only in the reference: {tracking['missing_keys']}",
        f"- Keys only in the live run: {tracking['added_keys']}",
        f"- Reference-only keys by track: {tracking['missing_by_track']}",
        f"- Live-only keys by track: {tracking['added_by_track']}",
        f"- Reference-only examples: {tracking['missing_examples']}",
        f"- Live-only examples: {tracking['added_examples']}",
        "",
        "Maximum absolute corner difference on shared keys:",
        "",
        _summary_line(tracking["max_corner"]),
        "",
        "IoU on shared keys:",
        "",
        _summary_line(tracking["iou"]),
        "",
        "Per-coordinate absolute differences:",
        "",
    ])
    for column, summary in tracking["coordinates"].items():
        lines.append(f"- `{column}`: {_summary_text(summary)}")
    lines.extend([
        "",
        "A low IoU on a shared key means that track ID does not refer to the same box.",
        "",
        "## Lane association",
        "",
        _agreement_line(comparison["lanes"], "Lane agreement"),
        "",
        _matrix_table(comparison["lanes"]["matrix"], LANE_STATUSES),
        "",
        "Definite means `IN_LANE` or `OUT_LANE`. Uncertain means `UNKNOWN`.",
        "",
        _count_line(
            "Definiteness changes",
            comparison["lanes"]["definiteness"],
        ),
        "",
        "## Motion estimation",
        "",
        f"- Finite growth in both: {comparison['motion']['both_finite']}",
        f"- Missing growth in both: {comparison['motion']['both_missing']}",
        (
            "- Missing only in the reference: "
            f"{comparison['motion']['reference_missing_only']}"
        ),
        (
            "- Missing only in the live run: "
            f"{comparison['motion']['new_missing_only']}"
        ),
        (
            "- Shared finite values on opposite sides of ±0.15/s: "
            f"{comparison['motion']['threshold_crossings']}"
        ),
        "",
        "Absolute `growth_robust_5` difference where both values are finite:",
        "",
        _summary_line(comparison["motion"]["growth"]),
        "",
        _agreement_line(
            comparison["motion"]["approach"],
            "Approach agreement",
        ),
        "",
        _matrix_table(
            comparison["motion"]["approach"]["matrix"],
            MOTION_STATES,
        ),
        "",
        _agreement_line(comparison["evidence"], "Evidence agreement"),
        "",
        "## Event stabilization",
        "",
        "Event-state agreement uses only shared observation keys, so a "
        "different set of tracks is not counted as a stabilizer change.",
        "",
        _agreement_line(comparison["event_states"], "Event-state agreement"),
        "",
        _matrix_table(comparison["event_states"]["matrix"], EVENT_STATES),
        "",
        "Full-file event-state counts, including keys that do not overlap:",
        "",
        _count_line(
            "Reference",
            comparison["distributions"]["reference"]["event_state"],
        ),
        _count_line("Live", new_dist["event_state"]),
        "",
        f"Reference episodes: `{reference_event_path}`",
        "",
        _episode_section(episodes),
        "",
        "## Performance",
        "",
        f"- Pipeline elapsed time: {timing['elapsed_seconds']} s",
        (
            "- Pipeline throughput: "
            f"{timing['frames_per_second']} frames/s"
        ),
        f"- Video rate: {video['fps']} frames/s",
        f"- Wall clock, including process startup: {wall_clock} s",
        f"- Annotated file size: {video_check.get('file_bytes')} bytes",
        "- Detector, lane, and writer times are not recorded separately.",
        _realtime_sentence(timing.get("frames_per_second"), video["fps"]),
        "",
        "## Findings",
        "",
    ])
    lines.extend(f"- {item}" for item in notes)
    lines.append("")
    return "\n".join(lines)


def findings(comparison: dict, metadata: dict) -> list[str]:
    notes = []
    tracking = comparison["tracking"]
    if tracking["missing_keys"] or tracking["added_keys"]:
        notes.append(
            "Some observation keys do not overlap. Differences for "
            "those keys come from the set of tracker identities, not "
            "from a second motion calculation on the same identity."
        )
    else:
        notes.append(
            "The observation keys match exactly. That still does not "
            "prove each track ID is the same physical object; the box "
            "differences are the check on the shared keys."
        )
    max_corner = tracking["max_corner"]["max"]
    if max_corner == 0:
        notes.append("Rounded boxes on every shared key are identical.")
    elif max_corner is not None and max_corner > 1.0:
        notes.append(
            "At least one shared track ID has a box difference above "
            f"1 pixel (maximum corner change {max_corner:.4g}). Those "
            "IDs should not be treated as the same detection."
        )
    episodes = comparison["episodes"]
    if (
        episodes["reference_count"] == episodes["new_count"]
        and len(episodes["exact"]) == episodes["reference_count"]
        and not episodes["identity_candidates"]
    ):
        notes.append(
            "Every completed episode matches its reference on track "
            "ID, start frame, confirmation frame, and end frame."
        )
    if episodes["shifted"] or episodes["split"] or episodes["merged"]:
        notes.append(
            "Same-track episode boundaries changed. Each shifted pair "
            "records whether the shared frames also changed boxes or "
            "lane labels."
        )
    if episodes["identity_candidates"]:
        notes.append(
            "Some episodes overlap in time under different track IDs. "
            "Those are identity candidates, not confirmed stabilizer "
            "changes and not confirmed physical matches."
        )
    crossings = comparison["motion"]["threshold_crossings"]
    if crossings and tracking["max_corner"]["max"] == 0:
        notes.append(
            f"{crossings} shared observation"
            f"{'s sit' if crossings != 1 else ' sits'} on opposite "
            "sides of ±0.15/s even though the rounded boxes match. "
            "Live growth uses the unrounded detector height. The "
            "threshold was not changed."
        )
    tau = episodes["exact_min_tau"]
    if tau["count"] and tau["max"] is not None and tau["max"] > 1e-9:
        notes.append(
            "Episodes with identical frames still differ in "
            "min_tau_proxy_s. Live growth uses unrounded detector "
            "coordinates; the Phase 6 file was produced from rounded "
            "CSV boxes. The values were not adjusted to match."
        )
    notes.append(
        "Live duration_s divides by the measured frame rate. "
        f"The Phase 6 file divides by {PHASE6_DURATION_DIVISOR}."
    )
    throughput = metadata["timing"].get("frames_per_second")
    video_fps = metadata["video"]["fps"]
    if (
        isinstance(throughput, (int, float))
        and isinstance(video_fps, (int, float))
        and throughput < video_fps
    ):
        notes.append(
            f"Throughput {throughput:.3f} frames/s is below the video "
            f"rate {float(video_fps):.5f} frames/s, so this run is not "
            "real time."
        )
    return notes


def _episode_section(episodes: dict) -> str:
    boundary = episodes["boundary_matches"]
    lines = [
        f"- Reference episodes: {episodes['reference_count']}",
        f"- Live episodes: {episodes['new_count']}",
        f"- Exact same-track matches: {len(episodes['exact'])}",
        f"- Same-track one-to-one pairs: {boundary['pairs']}",
        f"- Pairs with the same start frame: {boundary['start']}",
        (
            "- Pairs with the same confirmation frame: "
            f"{boundary['confirmation']}"
        ),
        f"- Pairs with the same end frame: {boundary['end']}",
        f"- Pairs with all three frames equal: {boundary['all_three']}",
        f"- Shifted same-track episodes: {len(episodes['shifted'])}",
        f"- Split episodes: {len(episodes['split'])}",
        f"- Merged episodes: {len(episodes['merged'])}",
        f"- Ambiguous same-track overlaps: {len(episodes['ambiguous'])}",
        (
            "- Reference episodes with no temporal counterpart: "
            f"{len(episodes['missing'])}"
        ),
        (
            "- Live episodes with no temporal counterpart: "
            f"{len(episodes['added'])}"
        ),
        (
            "- Cross-track temporal candidates: "
            f"{len(episodes['identity_candidates'])}"
        ),
        "",
        "Absolute min_tau_proxy_s difference on exact frame matches:",
        "",
        _summary_line(episodes["exact_min_tau"]),
        "",
        "### Reference episodes",
        "",
        _episode_table(episodes["reference_episodes_source"]),
        "",
        "### Live episodes",
        "",
        _episode_table(episodes["new_episodes_source"]),
    ]
    if episodes["shifted"]:
        lines.extend(["", "### Shifted same-track episodes", ""])
        for pair in episodes["shifted"]:
            lines.append(
                "- "
                f"{_episode_text(pair['reference'][0])} -> "
                f"{_episode_text(pair['new'][0])}; "
                f"{pair['attribution']}"
            )
    structural = (
        ("split", episodes["split"]),
        ("merged", episodes["merged"]),
        ("ambiguous", episodes["ambiguous"]),
    )
    if any(group for _, group in structural):
        lines.extend(["", "### Split, merged, or ambiguous", ""])
        for label, group in structural:
            for pair in group:
                lines.append(
                    f"- {label}: reference "
                    f"{_episode_list(pair['reference'])}; live "
                    f"{_episode_list(pair['new'])}"
                )
    if episodes["identity_candidates"]:
        lines.extend([
            "",
            "### Different track IDs with temporal overlap",
            "",
        ])
        for pair in episodes["identity_candidates"]:
            lines.append(
                "- candidate, not a physical match: "
                f"{_episode_text(pair['reference'])} overlaps "
                f"{_episode_text(pair['new'])} for "
                f"{pair['overlap_frames']} frames"
            )
    if episodes["missing"] or episodes["added"]:
        lines.extend(["", "### Episodes without a temporal counterpart", ""])
        for episode in episodes["missing"]:
            lines.append(f"- reference only: {_episode_text(episode)}")
        for episode in episodes["added"]:
            lines.append(f"- live only: {_episode_text(episode)}")
    return "\n".join(lines)


def _validate_video(video_path: Path, metadata: dict, source_video: Path | None):
    result = {
        "decoded_frames": None,
        "width": None,
        "height": None,
        "file_bytes": (
            video_path.stat().st_size if video_path.is_file() else None
        ),
        "sampled_mean_abs_diff": None,
        "problems": [],
    }
    if not video_path.is_file():
        result["problems"].append(f"annotated video is missing: {video_path}")
        return result
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        result["problems"].append(f"annotated video did not open: {video_path}")
        return result
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result["width"] = width
    result["height"] = height
    expected_width = int(metadata["video"]["width"])
    expected_height = int(metadata["video"]["height"])
    if width != expected_width or height != expected_height:
        result["problems"].append(
            f"annotated size {width}x{height} does not match "
            f"{expected_width}x{expected_height}"
        )
    expected_frames = int(metadata["video"]["frames_processed"])
    source = None
    if source_video is not None:
        if not source_video.is_file():
            result["problems"].append(
                f"source video is missing: {source_video}"
            )
        else:
            source = cv2.VideoCapture(str(source_video))
            if not source.isOpened():
                result["problems"].append(
                    f"source video did not open: {source_video}"
                )
                source.release()
                source = None
    sample_targets = {1, max(1, expected_frames // 2), expected_frames}
    diffs = []
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if frame.shape[1] != width or frame.shape[0] != height:
                result["problems"].append(
                    f"decoded frame {decoded} changed size"
                )
                break
            if source is None:
                continue
            ok_source, source_frame = source.read()
            if (
                decoded in sample_targets
                and ok_source
                and source_frame.shape == frame.shape
            ):
                difference = np.abs(
                    frame.astype(np.int16) - source_frame.astype(np.int16)
                )
                diffs.append(float(difference.mean()))
    finally:
        capture.release()
        if source is not None:
            source.release()
    result["decoded_frames"] = decoded
    if decoded != expected_frames:
        result["problems"].append(
            f"decoded {decoded} frames, metadata says {expected_frames}"
        )
    if diffs:
        result["sampled_mean_abs_diff"] = float(np.mean(diffs))
        if max(diffs) < 1.0:
            result["problems"].append(
                "sampled annotated frames are indistinguishable from the source"
            )
    return result


def _episode_problems(episode, fps, frames_processed, indexed):
    problems = []
    label = _episode_text(episode)
    ordered = (
        1
        <= episode.start_frame
        <= episode.confirmation_frame
        <= episode.end_frame
        <= frames_processed
    )
    if not ordered:
        problems.append(f"episode bounds are inconsistent: {label}")
    duration = episode.end_frame - episode.start_frame + 1
    if episode.duration_frames != duration:
        problems.append(f"episode duration_frames disagrees with bounds: {label}")
    if episode.positive_frames > episode.duration_frames:
        problems.append(f"episode has more positive frames than its duration: {label}")
    if episode.interruption_frames > episode.duration_frames:
        problems.append(
            f"episode has more interruption frames than its duration: {label}"
        )
    if episode.duration_s is not None:
        expected = episode.duration_frames / fps
        if abs(episode.duration_s - expected) > 1e-9:
            problems.append(
                f"episode duration_s does not use the measured FPS: {label}"
            )
    confirmed = False
    for frame in range(episode.start_frame, episode.end_frame + 1):
        row = indexed.get((frame, episode.track_id))
        if row is not None and str(row["event_state"]) == "CONFIRMED":
            confirmed = True
            break
    if not confirmed:
        problems.append(f"episode has no CONFIRMED observation: {label}")
    return problems


def _components(reference_episodes, new_episodes):
    reference_count = len(reference_episodes)
    total = reference_count + len(new_episodes)
    adjacent = {index: [] for index in range(total)}
    for reference_index, reference in enumerate(reference_episodes):
        for new_index, new in enumerate(new_episodes):
            if overlap_frames(reference, new) <= 0:
                continue
            new_node = reference_count + new_index
            adjacent[reference_index].append(new_node)
            adjacent[new_node].append(reference_index)
    seen = set()
    components = []
    for start in range(total):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        reference_group = []
        new_group = []
        while stack:
            index = stack.pop()
            if index < reference_count:
                reference_group.append(reference_episodes[index])
            else:
                new_group.append(new_episodes[index - reference_count])
            for nxt in adjacent[index]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append((reference_group, new_group))
    return components


def _identity_candidates(missing, added):
    pairs = []
    for reference in missing:
        for new in added:
            if reference.track_id == new.track_id:
                continue
            overlap = overlap_frames(reference, new)
            if overlap:
                pairs.append((overlap, reference, new))
    pairs.sort(
        key=lambda item: (
            -item[0],
            item[1].start_frame,
            item[2].track_id,
        )
    )
    used_reference = set()
    used_new = set()
    candidates = []
    for overlap, reference, new in pairs:
        if reference in used_reference or new in used_new:
            continue
        used_reference.add(reference)
        used_new.add(new)
        candidates.append({
            "reference": reference,
            "new": new,
            "overlap_frames": overlap,
        })
    unpaired_missing = [
        episode for episode in missing if episode not in used_reference
    ]
    unpaired_added = [
        episode for episode in added if episode not in used_new
    ]
    return candidates, unpaired_missing, unpaired_added


def _attribute_shift(reference, new, reference_index, new_index):
    start = min(reference.start_frame, new.start_frame)
    end = max(reference.end_frame, new.end_frame)
    compared = 0
    lane_mismatch = 0
    max_corner = 0.0
    for frame in range(start, end + 1):
        key = (frame, reference.track_id)
        if key not in reference_index or key not in new_index:
            continue
        compared += 1
        reference_row = reference_index[key]
        new_row = new_index[key]
        if str(reference_row["lane_status"]) != str(new_row["lane_status"]):
            lane_mismatch += 1
        for column in BOX_COLUMNS:
            max_corner = max(
                max_corner,
                abs(float(reference_row[column]) - float(new_row[column])),
            )
    if compared == 0:
        return "no shared observations in the combined frame span"
    if lane_mismatch:
        return (
            f"lane labels differ on {lane_mismatch} of "
            f"{compared} shared frames"
        )
    if max_corner > 1e-6:
        return (
            f"boxes differ by up to {max_corner:.4g} pixels "
            f"on {compared} shared frames"
        )
    return (
        "shared frames have the same rounded boxes and lane labels; "
        "the boundary change is consistent with growth from unrounded "
        "coordinates"
    )


def _boundary_record(reference_group, new_group):
    start_equal = False
    confirmation_equal = False
    end_equal = False
    if len(reference_group) == 1 and len(new_group) == 1:
        reference = reference_group[0]
        new = new_group[0]
        start_equal = reference.start_frame == new.start_frame
        confirmation_equal = (
            reference.confirmation_frame == new.confirmation_frame
        )
        end_equal = reference.end_frame == new.end_frame
    return {
        "reference": list(reference_group),
        "new": list(new_group),
        "start_equal": start_equal,
        "confirmation_equal": confirmation_equal,
        "end_equal": end_equal,
    }


def _group_tracks(episodes):
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode.track_id].append(episode)
    return grouped


def _definiteness_change(reference: str, new: str) -> str:
    reference_definite = reference in {"IN_LANE", "OUT_LANE"}
    new_definite = new in {"IN_LANE", "OUT_LANE"}
    if reference == new and reference_definite:
        return "unchanged_definite"
    if reference == new and reference == "UNKNOWN":
        return "unchanged_uncertain"
    if reference_definite and new == "UNKNOWN":
        return "definite_to_uncertain"
    if reference == "UNKNOWN" and new_definite:
        return "uncertain_to_definite"
    if reference_definite and new_definite:
        return "definite_label_changed"
    return "other"


def _track_ids(keys):
    return sorted({track for _, track in keys})


def _frame_count(keys):
    return len({frame for frame, _ in keys})


def _count_tracks(keys):
    return dict(Counter(track for _, track in keys))


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _growth_side(growth: float) -> str:
    if growth > 0.15:
        return "approaching"
    if growth < -0.15:
        return "receding"
    return "stable"


def _finite(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _hardware() -> str:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    text = completed.stdout.strip()
    return text or "unavailable"


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _summary_text(summary: dict) -> str:
    return (
        f"n={summary['count']}, mean={_fmt(summary['mean'])}, "
        f"median={_fmt(summary['median'])}, p95={_fmt(summary['p95'])}, "
        f"max={_fmt(summary['max'])}"
    )


def _summary_line(summary: dict) -> str:
    return f"- {_summary_text(summary)}"


def _count_line(label: str, counts: dict) -> str:
    if not counts:
        return f"- {label}: none"
    parts = [
        f"{key} {value}"
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    ]
    return f"- {label}: " + ", ".join(parts)


def _agreement_line(section: dict, label: str) -> str:
    agreement = section["agreement"]
    shown = "n/a" if agreement is None else f"{agreement:.6f}"
    return (
        f"- {label}: {section['matches']}/{section['compared']} ({shown})"
    )


def _matrix_table(matrix: dict, labels: tuple[str, ...]) -> str:
    header = "| reference \\ live | " + " | ".join(labels) + " |"
    separator = "| --- | " + " | ".join("---" for _ in labels) + " |"
    rows = []
    for reference in labels:
        cells = [str(matrix[reference][new]) for new in labels]
        rows.append(f"| {reference} | " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _episode_text(episode: Episode) -> str:
    return (
        f"track {episode.track_id} "
        f"{episode.start_frame}-{episode.end_frame} "
        f"(confirmation {episode.confirmation_frame})"
    )


def _episode_list(episodes: list[Episode]) -> str:
    if not episodes:
        return "none"
    return "; ".join(_episode_text(episode) for episode in episodes)


def _episode_table(episodes: list[Episode]) -> str:
    if not episodes:
        return "None."
    lines = [
        "| track | start | confirmation | end | positive | interruptions |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(
        episodes,
        key=lambda episode: (episode.start_frame, episode.track_id),
    )
    for episode in ordered:
        lines.append(
            f"| {episode.track_id} | {episode.start_frame} | "
            f"{episode.confirmation_frame} | {episode.end_frame} | "
            f"{episode.positive_frames} | {episode.interruption_frames} |"
        )
    return "\n".join(lines)


def _realtime_sentence(throughput, video_fps) -> str:
    if not isinstance(throughput, (int, float)) or not isinstance(
        video_fps,
        (int, float),
    ):
        return "- Real-time performance was not measured."
    if throughput >= video_fps:
        return (
            f"- Throughput {throughput:.3f} frames/s meets or exceeds "
            f"the video rate {float(video_fps):.5f} frames/s."
        )
    return (
        f"- Throughput {throughput:.3f} frames/s is below the video "
        f"rate {float(video_fps):.5f} frames/s. This run is not real time."
    )


if __name__ == "__main__":
    raise SystemExit(main())
