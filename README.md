# RoadVision

RoadVision is an experimental driving-perception prototype. From one forward-facing dashcam video it detects road users, tracks them, estimates the ego lane, and records image-space approach evidence.

The evidence is a research indicator. It is not a calibrated collision risk, a validated physical time to collision, or a safety-certified driver-assistance function. Do not use it to control a vehicle.

The current application is one command. It does not need the intermediate CSV files produced by the earlier experiments.

## Architecture

Each video run creates new detection, lane, and motion state. Nothing is carried over from a previous video.

```text
MP4
 |
 |-- lane detector, temporal filter, and smoother on every frame
 |
 |-- YOLO11n and ByteTrack on the original frame
 |
 |-- conservative vehicle-to-lane association
 |
 |-- per-track causal motion and approach evidence
 |
 |-- annotated frame, observation row, and, at the end, episodes
```

Lane perception is classical: blur, Canny edges, a road region of interest, Hough lines, slope filtering, and a validated lane polygon. A temporal filter rejects abrupt jumps. An exponential moving average then smooths accepted fits. The lane state is updated even when a frame has no tracked objects.

Detection uses pretrained YOLO11n. The retained classes are person, bicycle, car, motorcycle, bus, and truck (COCO 0, 1, 2, 3, 5, and 7). ByteTrack assigns IDs. The selected tracker file is `configs/bytetrack_threshold.yaml`.

Association compares the bottom-center of each box with the smoothed lane. `IN_LANE` and `OUT_LANE` are definite. `UNKNOWN` stays explicit when the lane is unavailable, the contact point is outside the fitted vertical range, or it lies within 5% of either boundary.

Approach evidence is computed online for each track:

1. Bounding-box height and time, using the measured frame rate.
2. A causal five-frame robust slope of log height. A missing frame clears the window.
3. `APPROACHING`, `STABLE`, `RECEDING`, or `UNKNOWN` around ±0.15/s.
4. An inverse-expansion proxy, `tau_proxy_s = 1 / growth`, only while growth is above 0.15/s.
5. Lane-aware evidence, then a stabilizer with three confirmations and a gap of at most two frames.

`CONFIRMED` means the stabilizer accepted an image-space pattern. `HELD` means an accepted pattern is being retained during a short interruption. Neither state is a collision warning.

Monocular distance and vanishing-point experiments remain in `scripts/`. They use a hypothetical camera and are not inputs to approach classification, so the application does not run them. The design is recorded in [Phase 8 integration](docs/phase8_integration.md).

## Installation

The project was developed and tested with Python 3.12. Commands below are run from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the YOLO11n weights into the repository root. The file is not committed.

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

Place a driving video at `data/videos/test_drive.mp4`, or pass another file with `--input`. Videos and `*.pt` weights are ignored by Git.

The default device is `mps`. On a machine without Apple Metal, pass `--device cpu` or `--device cuda`. If `mps` or `cuda` is requested and unavailable, the run stops. It does not switch to CPU.

## Run a video

Process the reference video:

```bash
python -m roadvision run \
    --input data/videos/test_drive.mp4 \
    --output runs/final
```

Process a different video on CPU:

```bash
python -m roadvision run \
    --input path/to/drive.mp4 \
    --output runs/custom \
    --device cpu
```

Use the checked-in configuration and stop after 100 frames:

```bash
python -m roadvision run \
    --config configs/roadvision.yaml \
    --input data/videos/test_drive.mp4 \
    --output runs/smoke \
    --max-frames 100
```

`--input`, `--output`, `--device`, and `--max-frames` override the loaded configuration in memory. They do not edit the YAML file. Omit `--config` to use the same built-in defaults.

A missing video or configuration file, an unavailable accelerator, or a video-writer failure prints `error: ...` and exits with status 1. Invalid arguments exit with status 2. Ordinary user errors do not print a traceback.

The earlier `scripts/run_integrated_video.py` entry point remains for the Phase 4 experiment. New runs should use `python -m roadvision`.

## Configuration

`configs/roadvision.yaml` repeats the built-in defaults.

| Setting | Default | Role |
|---|---|---|
| `video.input_path` | `data/videos/test_drive.mp4` | Input video |
| `video.output_dir` | `runs/final` | Output directory |
| `video.max_frames` | null | Process every frame when null |
| `detector.model_path` | `yolo11n.pt` | YOLO weights |
| `detector.device` | `mps` | `cpu`, `mps`, or `cuda` |
| `detector.confidence` | 0.25 | Detector confidence |
| `detector.classes` | 0, 1, 2, 3, 5, 7 | Road-user classes |
| `tracker.config_path` | `configs/bytetrack_threshold.yaml` | ByteTrack settings |
| `lane_filter.max_jump` | 100 | Temporal jump limit |
| `lane_filter.max_reference_age` | 5 | Reference lifetime |
| `lane_filter.max_candidate_gap` | 2 | Recovery gap |
| `lane_smoother.alpha` | 0.3 | Lane EMA weight |
| `association.lateral_margin` | 0.05 | Boundary uncertainty |
| `motion.robust_window` | 5 | Growth window |
| `approach.growth_threshold` | 0.15 | Approach deadband, 1/s |
| `time_to_contact.min_growth` | 0.15 | Minimum growth for the proxy |
| `stabilizer.min_confirmations` | 3 | Frames before confirmation |
| `stabilizer.max_gap` | 2 | Interruption tolerated while pending or held |

Paths are resolved from the working directory. Unknown YAML keys are rejected. The `lane_detector` values are locked to `src/lanes/lane_detector.py` because `detect_lane()` does not accept overrides. The two 0.15 settings are independent: one classifies approach, and the other gates the inverse-expansion proxy.

ByteTrack's own thresholds stay in its YAML file. `new_track_thresh` is 0.40 and `track_buffer` is 30. The detector confidence of 0.25 happens to equal `track_high_thresh`, but it is a separate setting.

## Outputs

The output directory receives four files.

`annotated.mp4` shows the lane polygon, track ID, class, and lane color. `IN_LANE` is yellow, `OUT_LANE` is green, and `UNKNOWN` is gray. Confirmed in-lane objects also show the approach label and the inverse-expansion proxy. A banner states that these are experimental image-space indicators, not collision warnings.

`observations.csv` has one row per tracked object. Coordinates are rounded to 2 decimals, confidence to 4, timestamp to 3, and the lateral margin to 5. `height`, `time_seconds`, growth, and the proxy use the unrounded detector box. The columns are:

`frame`, `timestamp`, `track_id`, `class_id`, `class_name`, `confidence`, `x1`, `y1`, `x2`, `y2`, `contact_x`, `contact_y`, `lane_status`, `lane_reason`, `baseline_status`, `baseline_reason`, `normalized_lateral_margin`, `lane_valid`, `lane_geometry_valid`, `lane_temporal_reason`, `height`, `time_seconds`, `growth_robust_5`, `approach_state`, `tau_proxy_s`, `evidence`, `event_state`.

`events.csv` lists completed evidence episodes: `track_id`, `start_frame`, `end_frame`, `confirmation_frame`, `duration_frames`, `duration_s`, `positive_frames`, `interruption_frames`, and `min_tau_proxy_s`. Zero episodes is a valid result.

`metadata.json` records the effective configuration, model, tracker settings, video properties, column lists, counts, and timing. Its schema version is `1`.

## Tests

```bash
python -m pytest tests
```

The suite covers configuration, lane logic, motion and risk rules, the per-track processor, pipeline resources, the command-line interface, and the regression comparison. Passing tests checks the implemented behavior. It does not certify driving performance.

These checks skip, rather than fail, when local reference artifacts are absent:

- `tests/test_track_replay.py` replays `runs/integrated_final.csv` against the Phase 6 evidence and event tables. It also needs `data/videos/test_drive.mp4` for the measured frame rate.
- `test_published_phase6_csv_contains_the_excluded_fields` reads `runs/phase6/final/phase6_results.csv`.
- `test_checked_in_runtime_files_when_present` checks the default video and `yolo11n.pt` only when those files are already present.

Those files are produced by the research workflow and are not part of a fresh clone. `runs/` is gitignored.

## Phase 6 regression

On the development checkout, replaying the stored rounded boxes reproduced the Phase 6 categories exactly: 2,656 observations, event counts of 2,559 `NONE`, 62 `PENDING`, 25 `CONFIRMED`, and 10 `HELD`, and the same 8 episode boundaries. Growth and the inverse-expansion proxy matched to `1e-9` because that replay used the rounded coordinates.

The live command was then compared with the published tables:

```bash
python -m roadvision run \
    --input data/videos/test_drive.mp4 \
    --output runs/full_regression

python scripts/compare_full_pipeline.py
```

The second command reads `runs/full_regression/` and the Phase 6 CSV files under `runs/phase6/`. It writes `runs/full_regression/comparison_report.md`. The verified live result was 2,656 shared observations, 68 track IDs, identical rounded boxes, identical event states, and 8/8 identical episode boundaries. One approach label differed at frame 470, track 2, because unrounded height moved growth across −0.15/s. The comparison and frame-rate conventions are documented in [Phase 8 integration](docs/phase8_integration.md).

## Phase 7 evaluation

Phase 7 audited one 1,728-frame recording. The labels are a small exploratory review, including corrections made after seeing predictions. They are not an independent benchmark. Details are in [Phase 7 evaluation](docs/phase7_evaluation.md).

| Area | Pilot result |
|---|---|
| Detection, 20 random frames | Precision 81.25%, recall 86.67%, F1 83.87%, mean matched IoU 0.8478 |
| Lane association, random matched objects | Strict accuracy 18.9%, definite coverage 27.0%, selective accuracy 70.0% |
| Tracking | 68 IDs, 2,656 observations, mean continuity 89.5%, 27 tracks with gaps |
| Approach episodes | 8 episodes on 4 tracks, mean confirmation delay 2.5 frames, none independently verified as physical approach |
| Sensitivity | The 0.15/s, 3-confirmation, gap-2 baseline was reproduced; 27 settings produced 4 to 13 episodes |

Most random-subset lane predictions were `UNKNOWN`, mainly because the contact point was above the fitted lane region.

## Earlier experiments

- [Tracking experiments](docs/tracking_experiments.md) selected the ByteTrack threshold file: 68 IDs and 2,656 observations on the reference video.
- [Association experiments](docs/association_experiments.md) selected the 5% lateral margin.
- [Phase 5 distance and vanishing point](docs/phase5_motion_estimation.md) is experimental and separate from the command.
- [Phase 6 approach evidence](docs/phase6_risk_assessment.md) defines the evidence states and the 8 published episodes.

## Reference-video measurements

The Phase 8 command processed the 1920×1080 reference video at its measured rate of 29.97002997002997 fps.

| Measurement | Result |
|---|---:|
| Frames | 1,728 |
| Pipeline time | 49.462 s |
| Wall clock, including startup | 50.82 s |
| Throughput | 34.936 frames/s |
| Machine | Apple M4, MPS |

That throughput is higher than this video's frame rate. It is an average over one file on one computer, not a camera-latency or deployment measurement. An earlier perception-only pipeline, without approach evidence, measured 39.30 frames/s. The Phase 1 detection-and-draw benchmark, excluding decode and file writing, measured about 75.5 frames/s. Those are different workloads.

On this recording the live pipeline also reported 1,111 temporally accepted lane frames (64.29% of 1,728), 301 `IN_LANE`, 571 `OUT_LANE`, and 1,784 `UNKNOWN` observations.

## Limitations

Lane geometry covers the near field. Distant vehicles often remain `UNKNOWN` because their contact point is above that region. Dashes, occlusion, and nearby vehicles still make the classical lane fit unstable.

Tracking IDs can break. On the reference video, 27 of 68 tracks contain gaps. A repeated ID is a tracker identity, not proof that the box belongs to the same physical object, although the live boxes matched the stored boxes on every shared key in this run.

Bounding-box growth is not physical approach. The inverse-expansion value is not a measured time to collision. Monocular distances elsewhere in the repository use a hypothetical camera. Confirmed episodes were not independently verified as vehicles approaching the camera.

The manual evaluation uses one short recording and a small, partly prediction-informed label set. Detection misses, class changes, and identity fragments remain possible. The 5% lane margin and the 0.15/s growth deadband are selected heuristics, not calibrated probabilities.

## Project status

Phases 0–8 are complete: detection, tracking, lane perception, association, experimental distance scripts, approach evidence, the Phase 7 audit, and the unified application. Phase 9 is the portfolio presentation. It is not part of this code.

RoadVision is an experimental research prototype for learning and demonstration. It is not autonomous-vehicle control software.
