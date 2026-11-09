# RoadVision

**Real-Time ADAS Perception System**

RoadVision is a computer vision project focused on developing a real-time automotive perception pipeline using forward-facing dashcam footage.

The system aims to detect and track surrounding road users, understand the ego lane, estimate relative motion and distance, and assess potential collision risks.

## Project Objectives

1. Detect vehicles, pedestrians, cyclists, and other relevant road users.
2. Track detected objects across consecutive frames.
3. Identify lane boundaries and the ego vehicle's driving lane.
4. Estimate distances to surrounding objects.
5. Estimate relative motion between the ego vehicle and tracked objects.
6. Calculate time-to-collision (TTC).
7. Visualize collision-risk warnings in real time.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Pipeline implementation |
| PyTorch | Deep-learning inference |
| Ultralytics YOLO11n | Object detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video processing and visualization |
| NumPy | Numerical operations |
| pandas | Tracking data analysis |

Additional technologies will be introduced as the project develops.

## Current Implementation

### Phase 1 — Object Detection

Implemented vehicle and pedestrian detection using a pretrained YOLO11n model.

The detector processes forward-facing driving footage, identifies relevant road-user classes, and generates an annotated output video.

**Initial benchmark:**

- Video resolution: 1920 × 1080
- Input frame rate: 30 FPS
- Total frames: 1,728
- Measured detection processing rate: approximately 75.5 FPS on Apple Silicon using MPS

The reported processing rate covers model prediction and visualization, excluding video decoding and output writing. It is not a full end-to-end latency measurement.

### Phase 2 — Multi-Object Tracking

Integrated ByteTrack to assign persistent IDs to detected vehicles across consecutive video frames.

The pipeline exports tracking results to CSV for quantitative evaluation.

#### Selected Tracking Configuration

Following a four-configuration ablation experiment, we selected the **threshold-only ByteTrack configuration** as our provisional baseline.

| Parameter | Value |
|---|---|
| Detector | YOLO11n |
| Tracker | ByteTrack |
| Detection confidence | 0.25 |
| New-track threshold | 0.40 |
| Track buffer | 30 frames |

#### Experimental Findings

Compared with the default ByteTrack configuration:

- Short-lived tracks decreased by **46.4%** (28 → 15).
- Unique track IDs decreased by **23.6%** (89 → 68).
- Mean track span increased from **1.22 to 1.47 seconds**.
- Tracked observations decreased by **5.1%**.
- Frame coverage decreased slightly from 80.32% to 79.75%.

The ablation experiment showed that increasing `new_track_thresh` accounted for the observed changes in the reported metrics, while increasing `track_buffer` from 30 to 45 frames produced no additional measurable effect.

These findings do not establish improved tracking accuracy because ground-truth object identities have not yet been evaluated.

For detailed methodology, results, and limitations, see the [Multi-Object Tracking Experiments](docs/tracking_experiments.md).

## Development Roadmap

- [x] Phase 0 — Project setup and environment configuration
- [x] Phase 1 — YOLO object detection baseline
- [x] Phase 2 — ByteTrack integration and ablation study
- [ ] Phase 2.4 — Alternative tracker comparison and further validation
- [ ] Phase 3 — Lane perception
- [ ] Phase 4 — Distance estimation
- [ ] Phase 5 — Relative motion estimation
- [ ] Phase 6 — Time-to-collision and collision-risk assessment
- [ ] Phase 7 — Real-time optimization and benchmarking

## Project Status

**In development — Object detection and multi-object tracking implemented.**

The current system is an experimental perception prototype and is not validated for safety-critical driving applications.