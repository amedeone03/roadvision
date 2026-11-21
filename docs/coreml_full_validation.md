# Full-video Core ML validation

Experimental check of the approved ONNX Runtime Core ML session on the 1,728-frame reference video. The production CLI, default PyTorch/MPS backend, thresholds, and algorithms were not changed.

## Command and environment

```bash
.venv-onnx/bin/python scripts/validate_coreml_full.py --run
```

Apple M4, macOS 26.6.2 arm64, Python 3.12.16, PyTorch 2.15.0, Ultralytics 8.4.193, ONNX Runtime 1.32.0. The provider options are `ModelFormat=MLProgram`, `MLComputeUnits=ALL`, `RequireStaticInputShapes=1`, `EnableOnSubgraphs=1`, `SpecializationStrategy=FastPrediction`, with `CPUExecutionProvider` registered as fallback. The override is installed only around each Core ML `run_video` call and restored afterward, including when that call raises.

Outputs are in ignored `runs/coreml_full_validation/`. `runs/full_regression` and `runs/profile_full` were not overwritten.

## Provider execution

A profiled probe of the same ONNX file assigned 320 of 320 graph nodes to one Core ML partition. The profile has 1 Core ML kernel and 0 CPU kernels. Both full-video sessions logged `CoreMLExecutionProvider` as the active provider. The input remains static `[1, 3, 640, 640]`.

## Throughput

| Run | Elapsed | Frames/s | Role |
| --- | ---: | ---: | --- |
| Historical PyTorch/MPS `runs/full_regression` | 49.462231 s | 34.935747 | Published baseline, not paired |
| Fresh PyTorch/MPS, this session | 46.510749 s | 37.152702 | Paired unprofiled run |
| Core ML, unprofiled | 40.193459 s | 42.992070 | Throughput result |
| Core ML, profiled | 38.370996 s | 45.034015 | Stage timings only |

Against the fresh PyTorch run, Core ML is **6.317 s faster, 1.157x end-to-end**. Against the historical baseline it is 9.269 s faster, 1.231x. The fresh PyTorch CSV matches `runs/full_regression` exactly, so the output differences below are not PyTorch run-to-run noise. The profiled clock is a second pass and is not the throughput number. Device `cpu` made zero GPU synchronization calls.

Steady profiled frames, first frame excluded, 1,727 frames: frame total mean 22.000 ms, median 20.769 ms, p95 31.593 ms, p99 34.529 ms. `inference_tracking` mean 9.479 ms, p95 18.539 ms. The published profiled PyTorch frame-total mean was 28.333 ms with MPS synchronization inside inference, so those stage figures are not a paired kernel comparison.

The first profiled frame took 353.158 ms, of which 330.319 ms was the inference call that compiles the Core ML model. That is 0.8% of the 40.193 s full run. A 100-frame clip still spends most of its time after compilation. A clip of only a few frames would be dominated by it.

The gain continues over the full video. It is a smaller fraction of the pipeline than the 2.70x `predict()` speedup, because lane detection and video writing do not use Core ML.

## Detections

Matched on rounded observation boxes, within class, one-to-one, at IoU >= 0.5. Track IDs were not required.

2,632 of 2,656 PyTorch observations matched 2,668 Core ML observations. 24 PyTorch boxes and 36 Core ML boxes were unmatched. 41 of 1,728 frames have a different count, and the largest count change is 2. Median IoU 0.997, 5th percentile 0.966, minimum 0.761. Median center shift 0.090 px, p95 1.342 px. Median absolute confidence gap 0.005.

Class counts: truck 1,211 versus 1,210, car 1,419 versus 1,429, bus 26 versus 29. The matched boxes are small numerical variations. The unmatched tail and the class-count shift are real, and they are a small share of the video.

## Tracking

67 of 68 PyTorch trajectories mapped to 69 Core ML trajectories. 65 of those mappings are numeric renumbering. 2 kept the same ID. 6 reference tracks overlapped more than one Core ML ID. 15 frames inside a mapped track preferred a different ID. Two mapped tracks differ in length by at least 5 observations: one long track is 6 observations longer on Core ML (165 shared frames, mean IoU 0.990), and one short track is 5 observations shorter (6 shared frames, mean IoU 0.807).

One PyTorch track has no geometric match: car track 93, frames 689–692, 4 observations. Two Core ML tracks have no match: truck track 12, frames 186–189, 2 observations, and bus track 144, frame 957, 1 observation. This is not an IDF1 or HOTA result.

## Lane, motion, and evidence

On the 2,632 matched observations:

| Field | Mismatches |
| --- | ---: |
| Lane status | 6 |
| Approach state | 115 |
| Evidence | 90 |
| Event state | 13 |

Where both sides have `tau_proxy_s`, the median absolute difference is 0.026, the 95th percentile is 0.558, and the maximum is 2.819. 1,604 matched observations are missing the value on one side, which is the usual blank proxy outside an approach. These values are experimental image-space motion, not physical collision risk.

## Episodes

Eight reference episodes and eight Core ML episodes. All eight windows match. None were added or dropped. Seven have the same start and end frames. The differences are:

| Reference | Core ML | Difference |
| --- | --- | --- |
| Track 2, frames 38–46 | Track 2, frames 38–46 | `min_tau_proxy_s` 2.980 → 2.578 |
| Track 2, frames 67–71 | Track 2, frames 67–71 | `min_tau_proxy_s` 1.055 → 1.044 |
| Track 10, frames 81–83 | Track 8, frames 81–83 | ID renumber; `min_tau_proxy_s` 0.648 → 0.642 |
| Track 2, frames 222–224 | Track 2, frames 222–224 | `min_tau_proxy_s` 2.156 → 2.144 |
| Track 2, frames 421–424 | Track 2, frames 421–423 | Ends one frame earlier; duration and positive frames 4 → 3 |
| Track 63, frames 579–582 | Track 64, frames 579–582 | ID renumber; `min_tau_proxy_s` 2.767 → 3.469 |
| Track 63, frames 586–599 | Track 64, frames 586–599 | Same bounds; positive frames 12 → 13, interruptions 2 → 1 |
| Track 66, frames 665–668 | Track 67, frames 665–668 | ID renumber; `min_tau_proxy_s` 1.766 → 1.841 |

## Tests

185 tests passed in the main environment. The suite does not need ONNX Runtime or the full video. New coverage checks frame-count deltas, category agreement despite renumbered IDs, episode windows with a changed proxy, and restoration of the original ONNX Runtime session after an exception.

## Recommendation

Keep PyTorch/MPS as the default. Core ML is fast enough, and stable enough across this recording, to justify a later opt-in Apple Silicon path: the full graph ran on Core ML, the end-to-end gain survived the full video, and all eight evidence episodes still appear. It is not ready to switch on. The frame 421 episode ends one frame early, one episode changes its interruption count, and approach state disagrees on 115 matched observations. Those differences should be accepted explicitly before the provider options become a production setting.
