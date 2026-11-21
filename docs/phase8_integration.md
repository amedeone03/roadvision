# Phase 8 — Integration and Regression

## Purpose

Phase 8 turns the research scripts into one reproducible video command. It does not replace the perception or approach algorithms, and it does not retune their thresholds.

```bash
python -m roadvision run \
    --input data/videos/test_drive.mp4 \
    --output runs/final
```

The command reads an MP4 directly. It does not require a previously generated CSV.

## Architecture and data flow

`roadvision/` is the application layer. The algorithms remain in `src/`.

```text
RoadVisionConfig
        |
        v
python -m roadvision run
        |
        v
run_video()
        |
        +-- new YOLO model and ByteTrack state
        +-- new TemporalLaneFilter and LaneSmoother
        +-- new TrackMotionProcessor
        |
        v
for each frame, in order:
        detect_lane()
        temporal filter and smoother
        YOLO track() on the original frame
        conservative lane association
        TrackMotionProcessor.update() for each track
        annotate and append one observation row
        |
        v
TrackMotionProcessor.finish()
        |
        +-- annotated.mp4
        +-- observations.csv
        +-- events.csv
        +-- metadata.json
```

The processor calls the existing functions for box height, Theil-Sen robust growth, approach classification, the inverse-expansion proxy, lane evidence, and event stabilization. A track keeps its own history. A frame gap clears that history before the new sample is added. `UNKNOWN` lane status and `HELD` event status keep their previous meanings.

Each execution builds its model, lane filters, and track state from scratch. Repeating a command does not reuse the previous run's temporal state.

## Why the pipeline is one causal pass

The Phase 6 research path saved each stage to CSV: integrated tracking, height and time, robust growth, approach class, the inverse-expansion proxy, lane evidence, and then the stabilizer. Every one of those stages uses only the current sample and earlier samples from the same uninterrupted track. A single forward pass can therefore apply them in that order while the video is being read.

The pass preserves the measured container frame rate, one-based frame numbers, the detector and tracker settings, the class filter, and the conservative association rule. Observation rows are written as frames are processed rather than collected into one in-memory table.

This is why a live run can agree with the offline chain. It is not because the outputs were forced to match.

## Why distance and vanishing point stay outside

Phase 5 estimates a road vanishing point, filters it over time, and converts image position to meters with a hypothetical camera. Those steps live in experiment scripts and add columns such as `filtered_vp_y` and `distance_dynamic_m`.

Approach classification does not read those columns. It reads robust log-height growth. Including Phase 5 in the command would add an unvalidated metric path and a second set of script-local filters without changing the published evidence decision. The application CSV therefore omits the Phase 5 fields. The scripts remain available for that separate experiment.

## CSV replay

Before running live detection again, the stored integrated-video boxes and lane labels were passed through `TrackMotionProcessor`. The inputs were `runs/integrated_final.csv` and the measured frame rate of `data/videos/test_drive.mp4`. The references were `runs/phase6/stabilized_evidence_results.csv` and `runs/phase6/stabilized_approach_events.csv`.

The replay matched:

- 2,656 observations joined on `(frame, track_id)`.
- Lane status, lane reason, baseline status, baseline reason, approach class, evidence, and event state exactly.
- Event counts: `NONE` 2,559, `PENDING` 62, `CONFIRMED` 25, `HELD` 10.
- Finite `growth_robust_5` and `tau_proxy_s` within `1e-9`, with missing values matching missing values.
- All 8 episodes on track ID, start, confirmation, end, duration in frames, positive frames, and interruption frames.

That close numeric match is expected. The replay used the same rounded boxes that had produced the Phase 6 tables. `duration_s` was not required to match: the processor divides by the measured frame rate, while the Phase 6 event file divides by the constant 29.97.

`tests/test_track_replay.py` repeats this check when those local files exist. A checkout without them skips the test.

## Full-video comparison

The live command was run on Apple M4 with the default MPS configuration:

```bash
python -m roadvision run \
    --input data/videos/test_drive.mp4 \
    --output runs/full_regression
```

`scripts/compare_full_pipeline.py` compared `runs/full_regression/observations.csv` and `events.csv` with `runs/phase6/final/phase6_results.csv` and `runs/phase6/stabilized_approach_events.csv`. It writes `runs/full_regression/comparison_report.md`.

The verified result:

| Check | Result |
|---|---|
| Frames | 1,728 at 1920×1080 |
| Observations | 2,656 |
| Track IDs | 68, with the same `(frame, track_id)` keys |
| Rounded boxes | Identical on all 2,656 shared keys |
| Lane labels | 2,656/2,656 |
| Evidence labels | 2,656/2,656 |
| Event states | 2,656/2,656 |
| Completed episodes | 8/8, including start, confirmation, and end |
| Approach labels | 2,655/2,656 |
| Automated tests | 136 passed |

A shared track ID is still a tracker identity. On this recording the identical rounded boxes mean the live and reference rows are the same stored detections, not an inference that every ID is a distinct physical object across the whole drive.

No threshold was changed to obtain this result.

## The frame 470, track 2 difference

One approach label differs.

| Field | Phase 6 reference | Live run |
|---|---|---|
| Frame, track | 470, track 2 | 470, track 2 |
| Rounded box | (881.80, 616.86)–(1031.10, 737.68) | same box |
| Growth | −0.150109/s | −0.14942/s |
| Approach | `RECEDING` | `STABLE` |
| Lane, evidence, event | `UNKNOWN`, `NO_APPROACH_EVIDENCE`, `NONE` | same three labels |

The rounded box height is 120.82 px. The live processor used the unrounded detector height, 120.8214111328125 px. That small change moved the five-frame slope from just below −0.15/s to just above it. `RECEDING` and `STABLE` both mean "not approaching," so the evidence and event labels do not change, and no episode boundary moves.

Where both growth values are finite, the absolute difference has mean 0.000557/s, median 0.000270/s, 95th percentile 0.00206/s, and maximum 0.00895/s. This is the only value that crosses a decision boundary.

## Rounded and unrounded boxes

The CSV stores `x1`, `y1`, `x2`, and `y2` to 2 decimals, matching the integrated-video export. `height`, `time_seconds`, `growth_robust_5`, and `tau_proxy_s` are computed from the full-precision detector box before that rounding.

The Phase 6 tables were computed from the already rounded CSV. Replaying those rounded values can match to `1e-9`. A live run can differ by a few thousandths of a reciprocal second, which matters only next to ±0.15/s. Metadata records `motion_uses_unrounded_detector_boxes: true`. The comparison script reports the difference and does not edit either file.

The same distinction appears in `min_tau_proxy_s` for the eight matching episodes: mean absolute difference 0.00233 s and maximum 0.00511 s.

## Frame-rate conventions

The reference container rate is 30000/1001, or 29.97002997002997 fps. The application measures that rate from the video and uses it for:

- `time_seconds = (frame - 1) / measured_fps`, with frame numbers starting at 1.
- The robust-growth time axis.
- Episode `duration_s = duration_frames / measured_fps`.

The CSV `timestamp` column is that same time rounded to 3 decimals. Classification does not use the constant 29.97. The Phase 6 event file uses 29.97 only when it converts a frame count to `duration_s`. OpenCV can report the written MP4 rate as 29.97 even though metadata keeps the measured rate. Motion is not recalculated from the written file.

## Performance and its limit

The full command on an Apple M4, using MPS, took 49.462 seconds inside the pipeline and 50.82 seconds of wall clock including process startup. Throughput was 34.936 frames/s. The video lasts about 57.66 seconds, so this run finished faster than the recording plays.

That supports faster-than-real-time processing for this file on this machine. It does not measure camera latency, dropped live frames, or another computer. The run does not time detection, lane fitting, tracking, and video writing separately, so it does not identify one stage as the bottleneck.

The annotated file for this run is 100,327,740 bytes. It decoded back to 1,728 frames at 1920×1080, and sampled frames differ from the source because of the overlay.

## Tests

`python -m pytest tests` passed 136 tests after the regression. The suite does not need the 1,728-frame video except for the optional replay and presence checks, which skip when their local files are absent.

## What Phase 8 does not claim

The integrated command reproduces the experimental evidence pipeline. It does not add physical validation. Confirmed episodes remain algorithmic image-space evidence, not verified approaches and not collision warnings.
