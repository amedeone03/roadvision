
# Vehicle-to-Lane Association — Phase 4

## Objective

Associate tracked road users with the estimated ego lane
while explicitly handling uncertain classifications.

## Pipeline

- YOLO11n object detection
- Threshold-tuned ByteTrack
- OpenCV lane detection
- Temporal geometric filtering
- EMA lane-boundary smoothing (alpha = 0.3)
- Vehicle-to-lane association

## Baseline Method

The original method classified the bottom-center of
each detected bounding box using point-in-polygon
testing.

Possible statuses:
- IN_LANE
- OUT_LANE
- UNKNOWN

UNKNOWN is returned when valid lane geometry is
unavailable or the contact point lies outside the
supported vertical interval.

## Problem Identified

Large trucks in adjacent lanes occasionally received
incorrect IN_LANE classifications.

Seven direct IN_LANE/OUT_LANE transitions occurred
across two truck tracks.

Inspection suggested that bounding-box contact points
were sensitive to changes in estimated box height.

## Experiments

### Contact-point EMA smoothing

Tested:
- Raw bottom-center
- EMA alpha = 0.3
- EMA alpha = 0.6

EMA smoothing did not reduce combined false IN_LANE
predictions in the two inspected truck windows.

The method was not adopted.

### Lateral boundary margin

Measured the horizontal distance between each
vehicle contact point and the two estimated lane
boundaries at the same image height.

Tested margins:
- 0%
- 5%
- 10%

A 5% normalized lateral margin was selected as
a conservative heuristic.

Points inside but within 5% of either lateral
boundary are classified UNKNOWN.

## Final Results

Input:
- 1,728 frames
- 1920 x 1080 resolution
- 29.97 FPS

Perception:
- 2,656 tracked observations
- 68 unique tracking IDs
- 1,111 temporally accepted lane frames
- 64.29% lane availability

Final association:
- IN_LANE: 301
- OUT_LANE: 571
- UNKNOWN: 1,784

Direct IN_LANE/OUT_LANE flips:
- Baseline: 7
- Final: 3
- Reduction: 57.1%

False IN_LANE classifications in two manually
inspected adjacent-lane truck windows:
- Baseline: 12
- Final: 6
- Reduction: 50%

The 5% margin changed seven observations from
IN_LANE to UNKNOWN.

## Visual Validation

24 selected observations were manually reviewed.

Both baseline and conservative classifiers agreed
with all 24 visual labels.

However, both classifiers made identical predictions
on these observations, so this sample does not
demonstrate superiority of the conservative method.

The sample includes repeated tracks and is not a
representative evaluation of all road conditions.

## Performance

Measured end-to-end offline video-processing
throughput: 39.30 FPS on the development machine.

This includes video reading, detection, tracking,
lane processing, association, visualization,
and output writing.

Live-camera latency has not been evaluated.

## Limitations

- Lane detection is limited to a near-field region.
- Bounding-box bottom-center is an approximate
  vehicle road-contact point.
- Large nearby vehicles can cause incorrect
  associations.
- The camera is not geometrically calibrated.
- The margin is heuristic, not a probability.
- Visual validation is limited to selected frames.
- The pipeline is a research prototype, not a
  safety-certified ADAS system.

## Conclusion

The conservative 5% margin reduced observed
classification instability and false positives
in selected diagnostic sequences.

The approach was retained for the final Phase 4
prototype, with UNKNOWN used to preserve uncertainty.
