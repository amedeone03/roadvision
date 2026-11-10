
# Phase 5 — Monocular Distance and Relative Motion Estimation

## Overview

Phase 5 extends RoadVision with experimental monocular distance
and relative-motion estimation.

The pipeline combines tracked vehicle observations, road vanishing-point
estimation, temporal filtering, and uncertainty screening.

Metric estimates are based on hypothetical camera parameters and are
not validated physical measurements.

## 1. Image-Space Motion

Vehicle motion is estimated from consecutive tracked bounding boxes.

Features include:

- Contact-point displacement in image coordinates
- Bounding-box height growth
- Logarithmic bounding-box height growth rate

To reduce noise, causal temporal smoothing methods were evaluated,
including linear regression and Theil–Sen estimation.

A five-frame robust estimator was retained as an experimental
calibration-free motion indicator.

## 2. Vanishing-Point Estimation

The road vanishing point is estimated from the intersection of the
left and right lane-line fits.

Three preprocessing approaches were compared:

1. Original lane detection
2. RGB object inpainting
3. Object-aware edge masking

Full-video results:

| Metric | Original | Inpainted | Edge-masked |
|---|---:|---:|---:|
| Valid VP frames | 1310 | 1080 | 1067 |
| Availability | 75.81% | 62.50% | 61.75% |
| Median vertical jump (px) | 2.87 | 2.57 | 2.17 |
| P95 vertical jump (px) | 63.28 | 34.29 | 14.11 |

Edge masking provided the strongest temporal consistency but
reduced estimate availability compared with the original detector.

## 3. Temporal Vanishing-Point Filtering

A causal temporal filter was evaluated using:

- EMA smoothing coefficient: 0.30
- Maximum reference jump: 25 px
- Sustained-shift confirmation: 3 observations
- Missing-data reset: 5 frames

Results:

| Metric | Original | Inpainted | Edge-masked |
|---|---:|---:|---:|
| Filtered valid frames | 1181 | 1007 | 1043 |
| P95 filtered vertical jump (px) | 3.89 | 3.33 | 2.64 |

The edge-masked approach was selected as the provisional method.

The filter preserved elevated road vanishing-point positions near
frames 650 and 850, although availability remained limited in some
regions.

## 4. Experimental Distance Estimation

Ground-plane distance was approximated using:

Z ≈ (H × fy) / (contact_y - VP_y)

Assumed camera parameters:

- Camera height: 1.5 m
- Vertical focal length: 1000 px
- Minimum horizon separation: 30 px

These are hypothetical parameters, not calibrated measurements.

Results:

- Vehicle observations: 2656
- Valid distance estimates: 1419 (53.43%)
- Missing VP observations: 1060
- Invalid horizon-separation observations: 177

Distance sensitivity was evaluated under ±10 px vanishing-point
perturbations.

Among valid central distance estimates:

- 1169 had complete uncertainty scenarios
- 250 had incomplete uncertainty scenarios
- 905 were flagged as high sensitivity

The 30 px minimum separation imposes an approximate 50 m maximum
reported range under the assumed camera parameters.

## 5. Relative Motion

Closing speed was defined as the negative temporal derivative of
estimated forward distance.

Three estimators were evaluated:

- Raw frame-to-frame differentiation
- Five-frame linear regression
- Nine-frame linear regression

| Metric | Raw | LR_5 | LR_9 |
|---|---:|---:|---:|
| Available estimates | 1142 | 603 | 220 |
| Median absolute closing speed (m/s) | 10.16 | 5.73 | 3.80 |

On 155 matched consecutive pairs, median frame-to-frame changes in
estimated closing speed were:

- Raw: 9.89 m/s
- LR_5: 2.16 m/s
- LR_9: 0.97 m/s

LR_5 was selected as a provisional compromise between temporal
stability and estimate availability.

## 6. Reliability Screening

A relative-motion estimate was classified as low-sensitivity only
when all five contributing distance observations satisfied the
chosen sensitivity criterion.

Results:

| Quality | Observations |
|---|---:|
| UNAVAILABLE | 2053 |
| UNCERTAIN | 503 |
| LOW_SENSITIVITY | 100 |

Among the 100 low-sensitivity observations:

- 77 were classified OUT_LANE
- 23 were classified UNKNOWN
- 0 were classified IN_LANE

These labels describe sensitivity under a chosen perturbation
scenario, not calibrated measurement confidence.

## 7. Integration

Final experimental outputs:

- runs/phase5/phase5_output.mp4
- runs/phase5/phase5_results.csv

The output video contains:

- Tracked vehicle bounding boxes
- Track identifiers
- Lane-relative classifications
- Experimental distance and closing-speed annotations when
  the low-sensitivity criterion is satisfied
- Explicit uncertainty indicators otherwise

Video properties:

- Frames: 1728
- Resolution: 1920 × 1080
- FPS: 29.97
- Unique tracks: 68
- Tracked observations: 2656

## 8. Validation

Structural video and CSV validation completed successfully.

Automated testing:

11 tests passed across 9 test files.

## 9. Limitations

- Camera intrinsics and mounting geometry are not calibrated.
- The simplified distance model assumes a locally planar road.
- Bounding-box contact points are imperfect proxies for ground contact.
- Vanishing-point availability is limited.
- Distance estimates are sensitive to horizon placement.
- Relative-speed estimates have not been validated against ground truth.
- No low-sensitivity metric motion estimates were obtained for
  IN_LANE observations under the selected criteria.
- The system is experimental and not suitable for safety-critical use.

## Conclusion

Phase 5 demonstrates an object-aware monocular distance and
relative-motion pipeline with causal temporal estimation and
explicit uncertainty screening.

The results support further research into camera calibration,
geometric robustness, and motion reliability before collision-risk
assessment can be treated as a validated capability.
