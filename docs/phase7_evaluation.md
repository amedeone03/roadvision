
# Phase 7 — Evaluation and Robustness

## Overview

This phase evaluates RoadVision's detection,
lane association, tracking continuity, and
experimental approach-evidence pipeline.

The evaluation combines a small manually annotated
dataset, quantitative diagnostics, visual inspection,
and parameter sensitivity experiments.

These results are exploratory and do not establish
safety-critical collision-detection performance.

## 1. Evaluation Dataset

Source: one 1,728-frame driving video
at approximately 29.97 FPS.

An evaluation manifest was generated containing
60 timeline-stratified random frames and
20 challenge frames.

For the pilot evaluation, 24 frames were manually
reviewed:

- 20 random-subset frames
- 4 challenge-subset frames

Manual annotations include object class, bounding
box, lane membership, and optional occlusion and
truncation flags.

The reviewed random frames are concentrated in
the earlier portion of the video.

Some annotations were corrected after inspecting
model predictions. Consequently, this is an
exploratory audit, not an independent held-out
benchmark.

## 2. Object Detection

Model: YOLO11n
Confidence threshold: 0.25
Input size: 640
Matching IoU threshold: 0.50

Results:

| Metric | Random | Challenge |
|---|---:|---:|
| Frames | 20 | 4 |
| TP | 39 | 10 |
| FP | 9 | 2 |
| FN | 6 | 1 |
| Precision | 0.8125 | 0.8333 |
| Recall | 0.8667 | 0.9091 |
| F1 | 0.8387 | 0.8696 |
| Mean matched IoU | 0.8478 | 0.7967 |

Precision, recall, and F1 are calculated using
class-aware, one-to-one bounding-box matching.

Mean IoU includes matched detections only.
It is not mean average precision (mAP).

Visual error inspection revealed omitted ground-truth
vehicles, especially oncoming traffic, highlighting
the importance of annotation completeness.

## 3. Lane Association

Lane predictions were matched with manually annotated
objects using class-aware IoU >= 0.50.

Random subset:

- Ground-truth objects: 45
- Matched objects: 37
- Strict lane accuracy: 18.9%
- Definite prediction coverage: 27.0%
- Selective accuracy: 70.0%

Challenge subset:

- Ground-truth objects: 11
- Matched objects: 9
- Strict lane accuracy: 66.7%
- Definite prediction coverage: 66.7%
- Selective accuracy: 100%

The random subset contains 27 UNKNOWN classifications
among 37 matched objects.

Across both subsets, 30 UNKNOWN predictions were
attributed to:

- Above lane coverage: 16
- Invalid lane geometry: 8
- Temporal jump rejection: 3
- Pending temporal recovery: 3

The main limitation is low classification coverage.

Three definite lane disagreements involve the same
tracked vehicle. Their ground-truth lane assignment
still requires independent confirmation.

The pilot includes some opposite-direction vehicles,
so the results do not constitute a clean benchmark
restricted to same-direction traffic.

## 4. Tracking Consistency

Tracking output:

- 2,656 observations
- 68 track IDs
- Mean within-track continuity ratio: 89.5%
- Median continuity ratio: 100%
- Tracks with internal gaps: 27
- Internal gaps: 122
- Missing frames within track spans: 352
- Short-lived tracks (up to 5 observations): 20
- Track IDs with class changes: 15

A heuristic identified 14 potential fragmentation
pairs.

Visual inspection supported fragmentation in
tracks 9 to 10 and 1 to 12, with probable
fragmentation in tracks 62 to 66.

These are visually assessed cases, not formal
identity-switch measurements.

No independent tracking identity annotations were
available for computing IDF1, HOTA, or MOTA.

## 5. Approach-Event Evaluation

The Phase 6 event stabilizer produced:

- 8 confirmed evidence episodes
- 4 unique track IDs
- Mean confirmation delay: 2.5 frames
- Median confirmation delay: 2 frames
- Maximum episode duration: 14 frames

Manual clip review:

| Criterion | Yes | No | Uncertain |
|---|---:|---:|---:|
| Apparent approach | 0 | 0 | 8 |
| Ego-lane correctness | 6 | 0 | 2 |
| Tracking stability | 4 | 0 | 4 |

All eight apparent-approach judgments remained
inconclusive.

This does not establish that eight detections
were false positives. It means that the available
visual evidence was insufficient for independent
verification of physical approach.

The review considered detected events only and
cannot establish event-level recall.

## 6. Parameter Sensitivity

The original Phase 6 baseline was reproduced exactly
before testing alternative configurations.

27 configurations were evaluated, varying:

- Growth threshold: 0.10, 0.15, 0.20 per second
- Minimum confirmations: 2, 3, 4
- Maximum gap: 0, 1, 2 frames

Baseline:

- Growth threshold: 0.15/s
- Minimum confirmations: 3
- Maximum gap: 2
- Confirmed episodes: 8

Changing growth threshold while holding other
parameters fixed produced 8, 8, and 7 episodes.

Changing required confirmations produced
12, 8, and 6 episodes.

Changing maximum gap produced 9, 8, and 8 episodes.

Across all 27 configurations, episode counts
ranged from 4 to 13.

The confirmation requirement has the strongest
observed effect on episode counts and confirmation
delay in the tested settings.

The baseline configuration is retained as an
experimental reference, not an empirically
validated optimum.

## 7. Overall Limitations

- Evaluation covers only one driving recording.
- The manually reviewed frame sample is small.
- Some labels were audited with model predictions.
- Lane associations have limited coverage.
- Tracking IDs can fragment or change class.
- Approach events lack independent physical-motion
  ground truth.
- No calibrated time-to-collision validation exists.
- Results cannot establish collision-risk accuracy.

## 8. Conclusion

RoadVision demonstrates an integrated and
reproducible experimental driving-perception pipeline.

Its object detection produces useful pilot metrics,
while lane association and tracking introduce
limitations that affect downstream approach evidence.

Sensitivity testing reproduces the baseline and
quantifies how temporal confirmation settings change
the system's outputs.

Further evaluation on independent recordings with
ground-truth vehicle trajectories would be required
for stronger performance claims.
