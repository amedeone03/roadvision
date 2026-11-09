# RoadVision

Real-Time ADAS Perception System

## Project Goal

Build a computer vision pipeline that processes
forward-facing driving footage to:

- Detect vehicles and pedestrians
- Track objects across frames
- Identify lane boundaries
- Estimate distance and relative motion
- Calculate time-to-collision (TTC)
- Visualize collision-risk warnings

## Technology Stack

Python, PyTorch, OpenCV, YOLO, NumPy

## Development

Phase 0: Project setup


## Experiments and Evaluation

RoadVision includes quantitative experiments to evaluate
the computer vision pipeline.

### Multi-Object Tracking

We implemented ByteTrack and evaluated four parameter
configurations using a controlled ablation study.

Increasing the new-track confidence threshold reduced
short-lived tracks by 46.4%, while decreasing tracked
observations by 5.1%.

These are proxy metrics and do not establish improved
identity-tracking accuracy.

See the [tracking experiments report](docs/tracking_experiments.md)
for the methodology, results, and limitations.