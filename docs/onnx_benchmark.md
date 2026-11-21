# ONNX Runtime benchmark

Experimental comparison of the existing `yolo11n.pt` weights with an ONNX Runtime CPU session. This is not a RoadVision backend. Thresholds, ByteTrack, and the production pipeline were left unchanged. The earlier inference-diagnostic stage times come from different passes and are not added to these measurements.

## Environment

Isolated `.venv-onnx`, created from the working Python 3.12.16 interpreter. The main `.venv` and `requirements.txt` were not changed.

| Package | Version |
| --- | --- |
| Python | 3.12.16 |
| PyTorch | 2.15.0 |
| Ultralytics | 8.4.193 |
| NumPy | 2.5.4 |
| ONNX | 1.23.2 |
| ONNX Runtime | 1.32.0 |

Available providers: `CoreMLExecutionProvider`, `AzureExecutionProvider`, `CPUExecutionProvider`. The benchmark session used `CPUExecutionProvider` only. There is no ONNX Runtime MPS provider.

## Export

`imgsz=640`, FP32, `dynamic=False`, `nms=False`, `simplify=False`, no quantization, no embedded tracker. Ultralytics selected ONNX opset 18.

Artifact: `runs/onnx_benchmark/yolo11n.onnx` (10,701,977 bytes, SHA-256 `0452dd0292885f40febb1d2a93ef438ef6ce0b3af0facde0fee41e09f0e75f4d`). Input `images` is `[1, 3, 640, 640]`. Output `output0` is `[1, 84, 8400]`, a raw detection head. NMS stays in Ultralytics. The file is gitignored.

## Detection

Same decoded frames, fresh models, confidence 0.25, IoU 0.7, classes 0, 1, 2, 3, 5, 7, `max_det=300`. Boxes are compared after Ultralytics postprocessing and before tracking. Matching is greedy, one-to-one, within class, at IoU >= 0.5.

| | PyTorch/MPS | ONNX Runtime CPU |
| --- | ---: | ---: |
| `predict()` boxes | 318 | 315 |
| Matched | 315 | |
| Unmatched | 3 reference, 0 candidate | |
| Median / minimum matched IoU | 0.998 / 0.909 | |
| Median / maximum center shift | 0.126 px / 6.319 px | |
| Median / maximum confidence gap | 0.0065 / 0.0587 | |

All 3 unmatched boxes were at or below confidence 0.30. Three matched pairs in that band had a confidence gap of at least 0.02. This is numerical agreement across engines, not an accuracy claim.

## Latency

Times are wall clock around one Ultralytics `predict()` or `track()` call, including preprocessing and Python NMS. Decoding and box extraction are outside the timer. PyTorch/MPS is synchronized before and after the call. ONNX CPU uses ordinary wall-clock time. The first frame of each fresh model is warmup and is excluded from the steady-state figures. 99 measured frames.

| | PyTorch/MPS | ONNX Runtime CPU |
| --- | ---: | ---: |
| Model load | 12.948 ms | 7.110 ms |
| First `predict()` | 3804.953 ms | 63.003 ms |
| Steady `predict()` mean / median / p95 / p99 | 13.450 / 11.748 / 21.023 / 23.921 ms | 17.254 / 16.877 / 19.609 / 20.740 ms |
| Steady `predict()` throughput | 74.348 frames/s | 57.958 frames/s |
| First `track()` | 408.789 ms | 40.885 ms |
| Steady `track()` mean | 12.530 ms | 17.563 ms |
| Steady `track()` throughput | 79.806 frames/s | 56.937 frames/s |

On this Apple M4, ONNX Runtime CPU `predict()` is slower than synchronized PyTorch/MPS by 3.804 ms of mean steady-state time. The ONNX cold start is much shorter. These figures are not MPS kernel times.

## Tracking

Fresh ByteTrack state, the current tracker file, `persist=True`.

| | PyTorch/MPS | ONNX Runtime CPU |
| --- | ---: | ---: |
| `track()` boxes | 278 | 280 |
| Matched at IoU >= 0.5 | 276 | |
| Median / minimum matched IoU | 0.997 / 0.775 | |
| Same track ID among matches | 194 of 276 | |
| Unique IDs / fragments | 6 / 8 | 6 / 8 |
| Frames with a different ID set | 51 | |

The box geometry stays close. The ID sets diverge because ByteTrack turns small score differences into different associations. The tracker configuration did not change, and equal IDs are not guaranteed.

## Pipeline spot check

The existing wrapper loaded the ONNX file through `detector.model_path` and `--device cpu`. No algorithm change was required. A separate 100-frame run wrote 278 observations and 3 episodes for PyTorch/MPS (3.814 s) and 280 observations and 3 episodes for ONNX CPU (4.421 s). Those durations include lane detection and video writing.

Episode windows agree: frames 38–46, 67–71, and 81–83, with the same duration, positive-frame, and interruption counts. The third episode is track 10 on PyTorch and track 8 on ONNX. `min_tau_proxy` differs with the box coordinates. A strict `(frame, track_id)` comparison therefore shows many missing and extra rows. Among 134 frame-and-class pairs that were unique on both sides, lane status agreed, approach state differed on 4, evidence on 1, and event state on 1.

## Reproduce

```bash
python -m venv .venv-onnx
.venv-onnx/bin/python -m pip install -r requirements.txt 'onnx>=1.12.0,<2.0.0' onnxruntime
.venv-onnx/bin/python scripts/benchmark_onnx.py --max-frames 100
```

Raw outputs stay in ignored `runs/onnx_benchmark/`. The regular test suite does not import ONNX Runtime or the exported file.

## Recommendation

Keep PyTorch/MPS as the Apple Silicon path. ONNX Runtime CPU is slower there in steady state, with detection agreement close enough that the 100-frame episode windows still matched. It is a reasonable optional CPU backend to try on Linux before any production flag. Do not add it to the default pipeline yet. Core ML was not tested.
