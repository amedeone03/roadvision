# Performance baseline

This record is the approved RoadVision v1.0 profiling baseline. It does not change detection, tracking, or event thresholds. The raw files remain in the ignored `runs/profile_full/` directory.

## Command

```bash
python -m roadvision run \
    --input data/videos/test_drive.mp4 \
    --output runs/profile_full \
    --profile
```

The device is the configuration default, `mps`. Profiling is off unless `--profile` is set. An ordinary run does not synchronize the GPU and does not write timing files.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M4 |
| Operating system | macOS 26.6.2, arm64 |
| Python | 3.12.16 |
| Torch | 2.15.0 |
| Ultralytics | 8.4.193 |
| Model | `yolo11n.pt` |
| Device | `mps` |
| Confidence | 0.25 |
| Classes | 0, 1, 2, 3, 5, 7 |
| Tracker | `configs/bytetrack_threshold.yaml` |
| Video | `data/videos/test_drive.mp4`, 1920×1080, 29.97003 FPS |
| Frames | 1,728 |

## Clocks

Stage timers use `time.perf_counter_ns` and do not overlap. `frame_total_ms` runs from the start of decoding through the observation flush. `other_ms` is the residual inside that interval.

`inference_tracking_ms` is one Ultralytics `track()` call, including the copy of boxes back to CPU. With profiling enabled, MPS is synchronized after that call and before the timer stops. The value is synchronized wall-clock time. It is not a GPU kernel duration, and YOLO is not split from ByteTrack.

The first completed frame is stored and excluded from every exported stage total. On this run it took 1391.941 ms, of which the inference and tracking stage took 1329.790 ms.

The pipeline clock starts with process work and stops after `events.csv`. It includes model construction, reader and writer setup, the frame loop, episode finalization, and `events.csv`. `metadata.json` and the profiling files are written after that clock stops. Pipeline frames per second are processed frames divided by that clock. They are not the reciprocal of mean `frame_total_ms`.

Synchronization can make a profiled run slower than an ordinary run.

## Measured frames

Statistics below use the 1,727 frames after warm-up.

| Stage | Total ms | Mean ms | Median ms | p95 ms | p99 ms | Share |
|---|---:|---:|---:|---:|---:|---:|
| Inference and tracking | 26383.343 | 15.277 | 13.873 | 23.863 | 26.540 | 53.9% |
| Lane detection | 10284.803 | 5.955 | 5.857 | 8.422 | 9.800 | 21.0% |
| Lane temporal filtering | 22.176 | 0.013 | 0.014 | 0.024 | 0.053 | 0.0% |
| Annotation | 1875.597 | 1.086 | 1.365 | 1.816 | 2.275 | 3.8% |
| Video writing | 8964.404 | 5.191 | 4.983 | 6.912 | 8.640 | 18.3% |
| Lane association | 15.415 | 0.009 | 0.006 | 0.025 | 0.044 | 0.0% |
| Motion and events | 108.469 | 0.063 | 0.061 | 0.127 | 0.210 | 0.2% |
| Decode | 1158.592 | 0.671 | 0.596 | 0.889 | 1.400 | 2.4% |
| CSV writing | 77.392 | 0.045 | 0.050 | 0.080 | 0.108 | 0.2% |
| Other | 41.123 | 0.024 | 0.022 | 0.043 | 0.058 | 0.1% |
| Frame total | 48931.314 | 28.333 | 27.363 | 38.581 | 44.209 | 100% |

Association and motion together are about 0.3% of measured frame time.

## Throughput

| Clock | Result |
|---|---|
| Profiled pipeline | 50.511512 s, 34.210023 frames/s |
| Steady measured frame | mean 28.333129 ms, 35.294 frames/s |
| Unprofiled `runs/full_regression` | 49.462231 s, 34.935747 frames/s |

The published v1.0 figure of 34.936 frames/s is that unprofiled rate rounded. The steady-state rate excludes the warm-up frame, startup, and finalization. Time outside the frame timers on the profiled run was 0.188 s.

## Behavior

The profiled outputs match `runs/full_regression`: 2,656 observation keys, rounded boxes, lane fields, approach state, evidence, event state, and all 8 episode boundaries. Accepted lane frames remain 1,111. No threshold was changed.
