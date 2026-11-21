# Inference diagnostics

This is the approved measurement of one `YOLO.track()` call. It does not change the RoadVision pipeline. Raw rows remain in the ignored `runs/inference_diagnostics/` directory.

The figures below come from different timing passes. They are not additive.

## Command

```bash
python scripts/benchmark_inference.py --max-frames 100
```

One hundred BGR frames are decoded before timing. Each pass constructs a new `yolo11n.pt` model and a new ByteTrack state. Settings match the application defaults: `imgsz` 640, confidence 0.25, IoU 0.7, classes 0, 1, 2, 3, 5, and 7, tracker `configs/bytetrack_threshold.yaml`, device `mps`.

## Environment

Apple M4, macOS 26.6.2 arm64, Python 3.12.16, Torch 2.15.0, Ultralytics 8.4.193.

## What was timed

`track_call_ms` is `time.perf_counter_ns` around `YOLO.track()` only. MPS is synchronized immediately before the call and immediately after it returns. Decoding, model construction, and the wrapper's later `.cpu()` copies are outside that interval.

The first completed frame is stored and excluded. On this run it took 1292.254 ms, while Ultralytics reported 4.856 ms of inference inside it. That long call is `model.warmup()`, which sits outside `Results.speed`.

`Results.speed` is Ultralytics' unsynchronized preprocess, inference, and postprocess split. On MPS those timers do not wait for the GPU. Their sum is not the synchronized `track()` time, and the gap between them and `track_call_ms` is not ByteTrack time.

`tracker_callback_ms` is recorded only when the ByteTrack `on_predict_postprocess_end` callback sits alone between two probes. The installed callback copies boxes to CPU and then calls `tracker.update`. It is empty on the first frame because that call registers the callback. The steady-state mean was 1.013 ms.

A benchmark-only predictor, unused by `python -m roadvision run`, synchronizes around preprocess, the forward pass, and postprocess.

## Steady state, 99 frames

| Measurement | Pass | Mean ms |
|---|---|---:|
| Synchronized `YOLO.track()` | Single synchronization | 14.129 |
| Synchronized forward pass | Per-stage synchronization | 13.101 |
| Synchronized preprocess | Per-stage synchronization | 1.586 |
| Synchronized postprocess | Per-stage synchronization | 2.903 |
| Tracker callback | Single synchronization | 1.013 |

Ultralytics' unsynchronized means on the single-synchronization pass ranked postprocess at 7.626 ms and inference at 3.234 ms. Synchronizing each method reversed that ranking. The unsynchronized postprocess interval is where unfinished MPS work becomes visible.

The per-stage pass had a higher mean `track()` time than the earlier single-synchronization pass. The two passes use separate models, and the second starts after the first has paid the long warmup, so that gap is not an isolated synchronization cost. An earlier run of the same script moved the gap in the other direction.

## What this does not justify

These numbers do not by themselves choose an export format. They do show that the synchronized forward pass was the largest measured piece of the call, and that ByteTrack's callback was about 1 ms. ONNX Runtime was left as the next experiment because the same Python NMS and ByteTrack path can load an exported model, including on Linux.
