
# RoadVision

**Real-Time ADAS Perception System**

RoadVision is a computer vision project focused on developing an automotive perception pipeline using forward-facing dashcam footage.

The system combines deep-learning-based object detection, multi-object tracking, classical lane detection, and geometric reasoning to identify surrounding road users and estimate their position relative to the ego vehicle's driving lane.

The long-term goal is to extend this perception pipeline with distance estimation, relative-motion analysis, time-to-collision (TTC) estimation, and collision-risk assessment.

> **Project status:** Phases 0–4 completed. Distance estimation, relative-motion estimation, and collision-risk assessment are planned.
>
> RoadVision is an experimental research prototype and is not designed or validated for safety-critical driving applications.

---

## Project Objectives

1. Detect vehicles, pedestrians, cyclists, and other relevant road users.
2. Track detected objects across consecutive video frames.
3. Detect lane boundaries and estimate the ego vehicle's driving lane.
4. Associate tracked road users with the estimated ego lane.
5. Estimate distances to surrounding vehicles.
6. Estimate relative motion between the ego vehicle and tracked objects.
7. Calculate time-to-collision (TTC).
8. Visualize potential collision risks.

Objectives 1–4 have been implemented for the current prototype.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Pipeline implementation |
| PyTorch | Deep-learning inference |
| Ultralytics YOLO11n | Object detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video processing, lane detection, geometric operations, and visualization |
| NumPy | Numerical computations |
| pandas | Tracking and perception-data analysis |
| Git / GitHub | Version control and experiment management |

The project uses PyTorch's Metal Performance Shaders (MPS) backend for GPU-accelerated inference on Apple Silicon.

---

## System Architecture

The integrated pipeline processes each video frame through two complementary perception branches.

```text
                    Dashcam Video
                         |
                     Video Frame
                         |
             +-----------+-----------+
             |                       |
             v                       v
        YOLO11n Detection      Lane Detection
             |                 (OpenCV + Hough)
             v                       |
        ByteTrack Tracking           v
             |                 Geometric Validation
             |                       |
             |                       v
             |                 Temporal Filtering
             |                       |
             |                       v
             |                 EMA Lane Smoothing
             |                       |
             +-----------+-----------+
                         |
                         v
              Vehicle-to-Lane Association
                         |
                         v
                IN_LANE / OUT_LANE
                     / UNKNOWN
                         |
              +----------+----------+
              |                     |
              v                     v
         Annotated Video        CSV Results
```

The two perception branches operate on the same video frames.

Object tracking maintains persistent identities across frames, while temporal lane filtering and smoothing reduce instability in estimated lane geometry.

---

## Current Implementation

### Phase 1 — Object Detection

Implemented road-user detection using a pretrained YOLO11n model.

The detector processes forward-facing driving footage and identifies relevant COCO classes:

- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Truck

Detected objects are visualized using bounding boxes and confidence scores.

#### Initial Detection Benchmark

| Metric | Result |
|---|---:|
| Video resolution | 1920 × 1080 |
| Input frame rate | 29.97 FPS |
| Total frames | 1,728 |
| Detection processing throughput | ~75.5 FPS |
| Inference device | Apple Silicon GPU (MPS) |

The initial benchmark covers model prediction and visualization, excluding video decoding and output writing.

It is not a full end-to-end latency measurement.

---

### Phase 2 — Multi-Object Tracking

Integrated ByteTrack to associate detections across consecutive frames and assign persistent tracking IDs.

Tracking observations are exported to CSV for quantitative analysis.

#### Selected Tracking Configuration

Following a four-configuration experiment, the threshold-tuned ByteTrack configuration was selected as the tracking baseline.

| Parameter | Value |
|---|---|
| Detector | YOLO11n |
| Tracker | ByteTrack |
| Detection confidence | 0.25 |
| New-track threshold | 0.40 |
| Track buffer | 30 frames |

Configuration file:

`configs/bytetrack_threshold.yaml`

#### Tracking Experiments

Compared with the default ByteTrack configuration:

- Short-lived tracks decreased by **46.4%** (28 → 15).
- Unique track IDs decreased by **23.6%** (89 → 68).
- Mean track span increased from **1.22 to 1.47 seconds**.
- Tracked observations decreased by **5.1%**.
- Frame coverage decreased slightly from **80.32% to 79.75%**.

The experiments indicated that changing the new-track threshold accounted for the observed differences in these metrics.

Increasing the track buffer from 30 to 45 frames produced no additional measurable improvement in the evaluated recording.

These measurements are tracking-behavior proxies, not ground-truth tracking accuracy metrics.

**Selected configuration results:**

| Metric | Result |
|---|---:|
| Unique tracking IDs | 68 |
| Tracked observations | 2,656 |
| Short-lived tracks | 15 |

For the detailed comparison and limitations, see [Multi-Object Tracking Experiments](docs/tracking_experiments.md).

---

### Phase 3 — Ego-Lane Perception

Implemented a classical computer vision pipeline for detecting the ego vehicle's lane boundaries.

The lane detector applies:

1. Grayscale conversion and Gaussian smoothing.
2. Canny edge detection.
3. Trapezoidal region-of-interest masking.
4. Probabilistic Hough line detection.
5. Segment filtering based on slope and image position.
6. Length-weighted linear fitting of left and right lane boundaries.
7. Geometric validation of the resulting lane polygon.

#### Geometric Validation

Candidate lane boundaries are checked for geometric plausibility, including:

- Correct left/right ordering.
- Valid image coordinates.
- Plausible lane widths.
- Non-crossing lane boundaries.

The fitted lane model is restricted to a near-field image region.

Lane estimates are rejected when their geometry does not satisfy the validation constraints.

#### Temporal Stabilization

Two additional components improve stability.

**TemporalLaneFilter**

Rejects abrupt changes in fitted lane coordinates and requires confirmation before accepting a new lane reference after prolonged instability.

**LaneSmoother**

Applies exponential moving average (EMA) smoothing to the accepted lane coordinates.

The selected smoothing parameter is:

`alpha = 0.3`

The temporal filter and EMA operate on lane geometry, not on vehicle detections.

#### Lane Perception Results

| Metric | Result |
|---|---:|
| Video frames | 1,728 |
| Temporally accepted frames | 1,111 |
| Lane availability | 64.29% |
| Longest unavailable interval | 38 frames |

The final temporal pipeline substantially reduced frame-to-frame variation in accepted lane coordinates.

However, the lane detector remains sensitive to dashed markings, occlusions, perspective, and nearby vehicles.

No ground-truth lane-segmentation dataset has been used to establish geometric detection accuracy.

---

### Phase 4 — Vehicle-to-Lane Association

Integrated object detection, tracking, and lane perception into a single video-processing pipeline.

For each tracked object, the system estimates its road-contact position using the bottom-center of its bounding box:

```python
contact_x = (x1 + x2) / 2
contact_y = y2
```

The contact point is compared against the estimated ego-lane geometry.

#### Association States

| Status | Interpretation |
|---|---|
| `IN_LANE` | Contact point is sufficiently inside the estimated ego lane |
| `OUT_LANE` | Contact point lies outside the estimated lateral lane boundaries |
| `UNKNOWN` | Available information is insufficient for a definite classification |

`UNKNOWN` can occur when:

- Lane geometry is unavailable.
- The contact point is above the supported lane region.
- The contact point is below the supported lane region.
- The contact point is too close to an estimated lane boundary.

This explicit uncertainty handling avoids forcing a lane classification when the geometric evidence is insufficient.

#### Conservative Lateral Association

The initial implementation used point-in-polygon testing.

Visual inspection revealed occasional incorrect `IN_LANE` classifications for large trucks traveling in adjacent lanes.

Further analysis showed that these errors were sensitive to bounding-box bottom-center coordinates and estimated lane boundaries.

The final implementation evaluates horizontal distances to the left and right lane boundaries at the vehicle's contact-point height.

A **5% normalized lateral uncertainty margin** is applied.

A vehicle inside the estimated lane but within 5% of either boundary is classified as `UNKNOWN`.

The 5% threshold is an experimentally selected heuristic, not a calibrated probability or automotive safety standard.

#### Association Experiments

Two approaches were evaluated:

**1. Contact-point smoothing**

Compared raw contact points against EMA smoothing with alpha values of 0.3 and 0.6.

Temporal smoothing did not consistently reduce false `IN_LANE` classifications in the inspected truck sequences and was therefore not adopted.

**2. Lateral uncertainty margins**

Compared 0%, 5%, and 10% normalized lateral margins.

The 5% margin offered a useful tradeoff between rejecting ambiguous truck observations and retaining classifications for a reference vehicle ahead.

#### Experimental Comparison

| Metric | Baseline | Final 5% margin |
|---|---:|---:|
| IN_LANE observations | 308 | 301 |
| OUT_LANE observations | 571 | 571 |
| UNKNOWN observations | 1,777 | 1,784 |
| Direct IN/OUT classification flips | 7 | 3 |
| False IN_LANE in inspected truck sequences | 12 | 6 |

The final method:

- Reduced direct lane-classification flips by **57.1%**.
- Reduced false `IN_LANE` observations by **50%** in two manually inspected adjacent-lane truck sequences.
- Changed seven classifications from `IN_LANE` to `UNKNOWN`.

These improvements are measured on the evaluated recording and selected diagnostic examples.

They do not establish general lane-association accuracy.

#### Manual Visual Validation

A separate set of 24 observations was selected for visual review.

Both the baseline and conservative methods agreed with all 24 visual labels.

However:

- Both methods produced identical predictions on this sample.
- Some vehicles appeared in multiple sampled frames.
- The sample was selected using predicted classifications.
- The evaluation did not cover a representative range of driving conditions.

Therefore, the result should not be interpreted as 100% generalization accuracy or as independent proof that the 5% method is superior.

For detailed experiments and limitations, see [Vehicle-to-Lane Association Experiments](docs/association_experiments.md).

---

## Integrated Pipeline Performance

The final integrated system combines:

- YOLO11n object detection.
- Threshold-tuned ByteTrack.
- Classical lane perception.
- Temporal lane validation.
- EMA lane smoothing.
- Conservative vehicle-to-lane association.
- Annotated video generation and CSV logging.

### Final Benchmark

| Metric | Result |
|---|---:|
| Input resolution | 1920 × 1080 |
| Input frame rate | 29.97 FPS |
| Processed frames | 1,728 |
| Unique track IDs | 68 |
| Tracked observations | 2,656 |
| Accepted lane frames | 1,111 |
| Lane availability | 64.29% |
| IN_LANE observations | 301 |
| OUT_LANE observations | 571 |
| UNKNOWN observations | 1,784 |
| End-to-end processing throughput | **39.30 FPS** |

The integrated pipeline processes the recording at approximately **1.31× its input frame rate** on the development machine.

The reported throughput includes video decoding, object detection, tracking, lane estimation, association, visualization, and output writing.

This supports faster-than-real-time offline processing for the tested recording.

It does not establish live-camera latency, deployment readiness, or performance across other hardware and driving environments.


### Phase 5 — Experimental Distance and Relative Motion

RoadVision includes an experimental monocular motion pipeline:

- Object-aware road vanishing-point estimation
- Causal temporal stabilization
- Approximate ground-plane distance estimation
- Five-frame relative-motion regression
- Geometric sensitivity screening
- Annotated video and CSV export

**Key results:** 1,728 frames processed, 68 tracked vehicles,
1,419 valid experimental distance observations, and 603
five-frame closing-speed estimates.

Only 100 motion estimates satisfied the selected full-window
sensitivity criterion.

**Limitation:** Metric estimates use hypothetical camera calibration
and are not validated physical measurements.

See [Phase 5 documentation](docs/phase5_motion_estimation.md).


### Phase 6 — Lane-Aware Approach Evidence

RoadVision includes an experimental approach-evidence
pipeline combining:

- Robust bounding-box scale-growth classification
- Inverse-expansion time-to-contact proxy
- Lane-aware approach classification
- Causal temporal event stabilization
- Annotated video and structured CSV export

**Results:** On 1,728 video frames, the pipeline
identified 1,066 apparent approach observations and
8 stabilized ego-lane approach episodes across 4 tracks.

All 39 automated tests passed, and the Phase 6
integration validator completed successfully.

**Limitation:** Detected episodes are not validated
collision threats. The inverse-expansion proxy must
not be interpreted as safety-certified TTC.

See [Phase 6 documentation](docs/phase6_risk_assessment.md).

---

## Running RoadVision

### 1. Clone the Repository

```bash
git clone https://github.com/amedeone03/roadvision.git
cd roadvision
```

### 2. Create a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Input Video

Place a dashcam recording at:

```text
data/videos/test_drive.mp4
```

The default configuration uses the Apple Silicon MPS backend.

For other supported hardware, update the `DEVICE` setting in `scripts/run_integrated_video.py`.

### 5. Run the Integrated Pipeline

```bash
python -m scripts.run_integrated_video
```

### 6. Inspect the Outputs

The final script generates:

```text
runs/
├── integrated_final_output.mp4
└── integrated_final.csv
```

The annotated video displays:

- Tracked-object IDs and classes.
- Bounding boxes and contact points.
- Accepted ego-lane geometry.
- Vehicle-to-lane association states.
- Lane availability.

The CSV contains per-object observations, including:

- Frame and timestamp.
- Tracking ID and object class.
- Detection confidence.
- Bounding-box coordinates.
- Vehicle contact point.
- Final lane status and reason.
- Original baseline classification.
- Normalized lateral margin.
- Geometric and temporal lane-validity information.

Generated videos and experiment outputs are excluded from version control.

---

## Testing

The project includes tests for the core lane-association and temporal-filtering logic.

Run them from the repository root:

```bash
python -m tests.test_vehicle_lane
python -m tests.test_vehicle_lane_detailed
python -m tests.test_vehicle_lane_conservative
python -m tests.test_temporal_filter
```

These tests cover:

- Geometric inside/outside classification.
- Explicit UNKNOWN states.
- Conservative boundary handling.
- Temporal rejection of abrupt lane changes.
- Recovery following unstable lane estimates.

Passing unit tests verifies expected component behavior but does not establish perception accuracy against real-world ground truth.

---

## Development Roadmap

- [x] Phase 0 — Project setup and environment configuration
- [x] Phase 1 — YOLO11n object detection
- [x] Phase 2 — ByteTrack integration and tracker comparison
- [x] Phase 3 — Ego-lane detection, validation, and temporal stabilization
- [x] Phase 4 — Vehicle-to-lane association and integrated video pipeline
- [ ] Phase 5 — Distance and relative-motion estimation
- [ ] Phase 6 — Time-to-collision and collision-risk assessment
- [ ] Phase 7 — Real-time optimization, extended validation, and benchmarking

### Planned Next Steps

**Phase 5 — Distance and Relative Motion**

Investigate image-based relative-motion measurements and the assumptions required for estimating physical distances from a monocular dashcam.

Metric distance estimates will require appropriate geometric assumptions or camera calibration.

**Phase 6 — Collision-Risk Assessment**

Investigate time-to-collision estimation and risk indicators based on tracked-object motion, with explicit handling of measurement uncertainty.

**Phase 7 — Optimization and Validation**

Evaluate processing latency, robustness across different recordings, and potential improvements to detection, tracking, and lane perception.

---

## Known Limitations

RoadVision currently has several important limitations:

1. **Near-field lane coverage:** The classical detector cannot reliably classify all distant vehicles.

2. **Approximate contact points:** Bounding-box bottom-centers do not always correspond to the actual road-contact position, particularly for large trucks.

3. **Uncalibrated geometry:** The system does not yet provide validated metric distance measurements.

4. **Detection and tracking errors:** Incorrect object classes, missed detections, and identity changes may occur.

5. **Limited evaluation:** Experiments use a short driving recording and selected visual examples.

6. **Heuristic uncertainty:** The 5% lateral margin is not a calibrated confidence estimate.

7. **Live deployment:** Real-world camera latency and automotive deployment constraints have not been validated.

RoadVision is intended for experimentation, learning, and portfolio demonstration, not autonomous vehicle control or safety-critical decision-making.

---

## Project Status

**Phase 4 complete — Integrated ADAS perception prototype operational.**

The current system detects and tracks road users, estimates the ego lane, and associates tracked objects with lane geometry while explicitly representing uncertain classifications.

The next development phase focuses on distance and relative-motion estimation.
