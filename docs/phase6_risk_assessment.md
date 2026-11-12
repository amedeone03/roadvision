
# Phase 6 — Lane-Aware Approach Evidence

## Overview

Phase 6 extends RoadVision with experimental
approach detection, inverse-expansion time estimation,
lane-aware evidence classification, and temporal
event stabilization.

The goal is to identify and review apparent vehicle
approach episodes, not to issue validated collision
warnings.

The pipeline reuses the tracking, lane association,
and robust motion features from earlier phases.

## 1. Approach Detection

Apparent scale motion is estimated using the
five-frame robust logarithmic bounding-box height
growth rate:

g = d(log(h)) / dt

where h is the tracked object's bounding-box height.

An experimental deadband of 0.15/s classifies
observations into:

- APPROACHING
- STABLE
- RECEDING
- UNKNOWN

Results over 2,656 tracked observations:

| State | Observations |
|---|---:|
| APPROACHING | 1066 |
| STABLE | 694 |
| RECEDING | 327 |
| UNKNOWN | 569 |

Thresholds of 0.10/s and 0.20/s were also evaluated.

Compared with the 0.15/s baseline, these changed
187 and 157 classifications respectively.

These classifications describe apparent image-scale
motion, not physical relative velocity.

## 2. Experimental Time-to-Contact Proxy

For a fixed-size object under simplified pinhole
projection geometry:

h is approximately proportional to 1/Z.

This gives the inverse-expansion relationship:

tau_proxy = 1/g

for sufficiently positive growth g.

The proxy is calculated only when g > 0.15/s.

Results:

- Available estimates: 1066
- Median proxy: 1.866 seconds
- Estimates below 1 second: 329
- Estimates below 2 seconds: 572
- Estimates below 3 seconds: 753

The inverse-expansion time is not a validated
physical time-to-collision measurement.

It may be affected by changes in object orientation,
bounding-box geometry, occlusion, camera movement,
and tracking errors.

## 3. Lane-Aware Evidence

Approach states are combined with existing
vehicle-to-lane classifications.

The evidence categories are:

- EGO_LANE_APPROACH
- OTHER_LANE_APPROACH
- LANE_UNCERTAIN_APPROACH
- NO_APPROACH_EVIDENCE
- MOTION_UNAVAILABLE

Results:

| Evidence category | Observations |
|---|---:|
| EGO_LANE_APPROACH | 60 |
| OTHER_LANE_APPROACH | 335 |
| LANE_UNCERTAIN_APPROACH | 671 |
| NO_APPROACH_EVIDENCE | 1021 |
| MOTION_UNAVAILABLE | 569 |

Only 60 observations across six tracks showed
simultaneous apparent approach and confirmed
ego-lane association.

## 4. Temporal Event Stabilization

An initial detector required three consecutive
ego-lane approach observations and identified
nine episodes.

A causal state machine was then implemented with:

- Three positive observations for confirmation
- Up to two consecutive nonpositive interruptions
- Immediate termination on recession or OUT_LANE
- Reset on missing track frames

The state machine distinguishes:

- NONE
- PENDING
- CONFIRMED
- HELD

HELD retains event context but does not imply
current confirmed approach evidence.

## 5. Event Stabilization Results

The stabilizer identified eight confirmed episodes
across four tracked vehicles.

| State | Observations |
|---|---:|
| NONE | 2559 |
| PENDING | 62 |
| CONFIRMED | 25 |
| HELD | 10 |

The longest episode lasted 14 frames,
approximately 0.47 seconds.

In Track 63, brief interruptions were bridged,
combining previously fragmented observations from
frames 586–599 into one confirmed episode.

Episode start frames are recorded retrospectively.
Online confirmation occurs only when sufficient
positive evidence has accumulated.

## 6. Video Integration

Phase 6 outputs:

- runs/phase6/final/phase6_output.mp4
- runs/phase6/final/phase6_results.csv
- runs/phase6/stabilized_approach_events.csv

The annotated video includes:

- Vehicle bounding boxes and track IDs
- Lane association
- Apparent motion classifications
- Experimental inverse-expansion time
- Temporally stabilized evidence states

No safety-critical collision warning is generated.

## 7. Validation

The video contains:

- 1728 frames
- Resolution: 1920 x 1080
- Frame rate: 29.97 FPS
- 2656 tracked observations
- 68 unique tracks

Integration validation passed.

All 39 automated tests passed.

The tests cover individual components and selected
state-machine behaviors. They do not establish
real-world collision-detection accuracy.

## 8. Limitations

- Evaluation uses one short driving recording.
- Approach estimates rely on bounding-box scale.
- The inverse-expansion proxy is not ground-truth TTC.
- Lane association can be unavailable or incorrect.
- Short evidence episodes can reflect perception noise.
- Thresholds are experimental, not safety-calibrated.
- Temporal confirmation introduces detection delay.
- No manually labeled collision-risk benchmark is available.
- Confirmed approach evidence does not prove a collision course.

## Conclusion

Phase 6 adds explainable, uncertainty-aware
approach-evidence analysis to RoadVision.

The system can identify and stabilize apparent
approach episodes while preserving distinctions
between confirmed motion evidence, uncertainty,
and unavailable observations.

Further evaluation with independent ground-truth
annotations and additional driving scenarios is
required before collision-risk performance can
be assessed.
