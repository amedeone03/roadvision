# Multi-Object Tracking: Design and Experimental Evaluation

## 1. Objective

RoadVision aims to develop a camera-based automotive perception pipeline capable of detecting and tracking surrounding road users, identifying the ego lane, estimating relative motion, and assessing potential collision risks.

After implementing object detection, the next challenge was maintaining consistent object identities across consecutive video frames.

This is essential because downstream tasks such as relative-motion and time-to-collision estimation require observations of the same physical object over time.

## 2. Model Selection

### Object Detection: YOLO11n

We selected the pretrained YOLO11 nano model (`yolo11n.pt`) as our initial object detector.

The main reasons were:

- **Computational efficiency:** The nano architecture is designed for relatively lightweight inference, making it suitable for an initial real-time perception prototype.
- **Pretrained object classes:** The model supports relevant COCO classes, including cars, trucks, buses, motorcycles, bicycles, and pedestrians.
- **Ease of integration:** The Ultralytics framework provides a straightforward interface for video inference and multi-object tracking.
- **Hardware compatibility:** Inference can use PyTorch's MPS backend on Apple Silicon.

We initially used a detection confidence threshold of 0.35, then reduced it to 0.25 for tracking experiments to retain more candidate detections.

### Multi-Object Tracking: ByteTrack

Object detection alone does not maintain identities between frames. We therefore integrated ByteTrack to associate detections over time.

ByteTrack uses bounding-box association and motion prediction, including lower-confidence detections in a secondary association stage.

We selected it as the initial tracker because it integrates directly with Ultralytics and provides a practical baseline without requiring training an additional appearance-embedding model.

Its limitations include possible identity fragmentation under occlusion, missed detections, and camera motion.

## 3. Experimental Setup

The experiments used:

| Component | Configuration |
|---|---|
| Input | Forward-facing highway driving footage |
| Resolution | 1920 × 1080 |
| Video frame rate | 30 FPS |
| Total frames | 1,728 |
| Duration | 57.6 seconds |
| Detector | YOLO11n |
| Tracker | ByteTrack |
| Tracking detection confidence | 0.25 |
| Hardware | Apple Silicon Mac, MPS backend |

Tracking results were exported to CSV, including frame number, timestamp, tracking ID, object class, confidence score, and bounding-box coordinates.

All configurations were evaluated using the same video and detector.

## 4. Parameter Tuning and Ablation Study

We investigated two ByteTrack parameters:

**`new_track_thresh`:** The confidence threshold used when initializing new tracks. Increasing this value may reduce short-lived or unreliable track creation, but may also delay or prevent tracks for genuine low-confidence objects.

**`track_buffer`:** The number of frames for which lost tracks are retained. Increasing this value may allow longer interruptions before a track is removed, although it does not guarantee successful reassociation.

Four experiments were conducted:

| Configuration | `new_track_thresh` | `track_buffer` |
|---|---:|---:|
| Baseline | 0.25 | 30 |
| Threshold only | 0.40 | 30 |
| Buffer only | 0.25 | 45 |
| Both tuned | 0.40 | 45 |

All other tracker settings were kept consistent.

## 5. Experimental Results

| Metric | Baseline | Threshold only | Buffer only | Both tuned |
|---|---:|---:|---:|---:|
| Unique track IDs | 89 | 68 | 89 | 68 |
| Tracked observations | 2,800 | 2,656 | 2,800 | 2,656 |
| Short tracks (fewer than 5 observed frames) | 28 | 15 | 28 | 15 |
| Short-track percentage | 31.46% | 22.06% | 31.46% | 22.06% |
| Mean track span (s) | 1.221 | 1.475 | 1.221 | 1.475 |
| Median track span (s) | 0.433 | 0.483 | 0.433 | 0.483 |
| Frame coverage | 80.32% | 79.75% | 80.32% | 79.75% |

Track span is defined as the elapsed interval between the first and last observed frame of a tracking ID, including any gaps.

Frame coverage represents the percentage of video frames containing at least one tracked object.

### 5.1 Effect of the New-Track Threshold

Increasing `new_track_thresh` from 0.25 to 0.40 resulted in:

- 23.6% fewer unique track IDs.
- 46.4% fewer short-lived tracks.
- 20.7% higher mean track span.
- 5.1% fewer tracked observations.
- A decrease in frame coverage from 80.32% to 79.75%.

These results suggest that stricter track initialization reduces short-lived tracks.

However, increasing the threshold may also suppress valid low-confidence objects. Furthermore, removing short tracks from the population can increase average track duration without improving the continuity of existing tracks.

### 5.2 Effect of Track Buffer

Increasing `track_buffer` from 30 to 45 frames produced no change in the reported aggregate metrics.

The buffer-only experiment matched the baseline, while the fully tuned configuration matched the threshold-only experiment.

For this particular recording, increasing the buffer therefore provided no additional measurable benefit according to the selected metrics.

This does not imply that the parameter is generally ineffective.

### 5.3 Long-Term Tracking

The longest track lasted approximately 18.13 seconds in all four configurations.

In the threshold-only configuration, this track was observed in 539 of its 544 spanned frames, indicating strong continuity for that object.

However, other tracks exhibited substantial gaps, showing that identity continuity remains an issue worth investigating.

## 6. Limitations

The current evaluation relies on proxy metrics rather than annotated ground-truth object identities.

In particular:

- Fewer tracking IDs do not necessarily indicate fewer identity switches.
- Short tracks may represent real vehicles briefly visible in the camera.
- Longer average track spans may partly result from excluding short tracks.
- Frame coverage does not measure whether every visible object was successfully tracked.
- The experiment used a single highway video and may not generalize to urban scenes.
- The effects of `new_track_thresh` and `track_buffer` were evaluated using aggregate statistics, not formal tracking-accuracy metrics.

A more rigorous evaluation would include manually annotated trajectories or a benchmark dataset, allowing ID-switch counts, IDF1, and HOTA to be measured.

## 7. Conclusion and Next Steps

The ablation study demonstrated that increasing the new-track confidence threshold was responsible for the observed changes in our tracking metrics, while increasing the track buffer produced no additional measurable effect.

We selected the threshold-only configuration (`new_track_thresh=0.40`, `track_buffer=30`) as a **provisional candidate** because it produced fewer short-lived tracks without requiring the longer buffer.

This choice is not yet validated as the most accurate configuration for an ADAS system, since it also reduced the number of tracked observations.

The next steps are to compare ByteTrack with BoT-SORT, evaluate identity continuity visually or against annotated trajectories, and benchmark processing latency before integrating lane perception.