# Optional Core ML backend

PyTorch on MPS is the RoadVision default. `onnx-coreml` is an optional experimental backend for Apple Silicon. It runs the existing FP32 static ONNX export through ONNX Runtime's `CoreMLExecutionProvider`. It does not change detection thresholds, ByteTrack, lane association, motion, event stabilization, or the four output schemas.

The labels in `observations.csv` and `events.csv` remain algorithmic image-space evidence. They are not validated collision warnings and they are not physical time-to-collision estimates.

## When to use it

Use `pytorch` for the default path and for any result you want to compare with the published PyTorch/MPS recording.

Use `onnx-coreml` on a Mac when you want the faster ONNX Runtime Core ML session and can accept the numerical differences below. It requires macOS, an ONNX Runtime build that advertises `CoreMLExecutionProvider`, and a model file you supply. A missing dependency, a non-Mac host, a PyTorch `.pt` path, or `--device mps` stops the run. RoadVision does not switch back to PyTorch.

## Environment

Normal installs do not need ONNX Runtime. The experimental session was measured in an isolated environment:

```bash
python -m venv .venv-onnx
.venv-onnx/bin/python -m pip install -r requirements.txt 'onnx>=1.12.0,<2.0.0' onnxruntime
```

The verified stack was Python 3.12.16, PyTorch 2.15.0, Ultralytics 8.4.193, ONNX 1.23.2, and ONNX Runtime 1.32.0. `.venv-onnx/` is gitignored.

## ONNX model

Export is a manual step. The video command does not download or export weights.

```bash
.venv-onnx/bin/python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11n.pt").export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    nms=False,
    simplify=False,
    device="cpu",
)
PY
```

That writes FP32 `yolo11n.onnx` with a static `1x3x640x640` input and NMS left outside the graph. Keep the file outside Git. `*.onnx` is ignored.

Ultralytics selects its own providers for an ONNX file, and its MPS path uses the NeuralNetwork format. Its public loader does not accept execution-provider options. RoadVision replaces `onnxruntime.InferenceSession` only from model construction through the first frame, then restores it. That attribute is process-wide. A second Core ML run that overlaps this window stops with an error. Run videos one at a time. Sequential runs each receive a new model and a new ByteTrack state. The verified provider list is:

- `CoreMLExecutionProvider`
- `CPUExecutionProvider` as fallback
- `ModelFormat=MLProgram`
- `MLComputeUnits=ALL`
- `RequireStaticInputShapes=1`
- `EnableOnSubgraphs=1`
- `SpecializationStrategy=FastPrediction`

`metadata.json` records the providers and options the session actually returned. A run does not repeat the graph-node profile. On the benchmarked Apple M4 that profile assigned all 320 nodes to one Core ML partition and left no CPU kernels.

## Command

```bash
.venv-onnx/bin/python -m roadvision run \
  --backend onnx-coreml \
  --model path/to/yolo11n.onnx \
  --device cpu \
  --input path/to/video.mp4 \
  --output runs/coreml_demo \
  --max-frames 100
```

Existing commands that omit `--backend` stay on PyTorch, including the checked-in YAML. Each run still builds a new model and a new ByteTrack state.

## Measured differences

On the Apple M4 full recording, a paired unprofiled PyTorch/MPS run took 46.510749 s (37.152702 frames/s). The Core ML session took 40.193459 s (42.992070 frames/s), about 15.7% higher end-to-end throughput. The first frame includes roughly 0.33 s of compilation. The historical 49.462 s baseline was not paired with that run.

2,632 of 2,656 reference observations matched geometrically, with median IoU 0.997. All eight evidence episodes were recovered. One episode ends one frame earlier, one keeps its window but changes an interruption count, and 115 matched observations differ in approach state. Track IDs are not interchangeable between backends. Those differences are backend numerical behavior, not a retune, and they are not a claim of better tracking or safer warnings.

Details are in [the Core ML provider benchmark](coreml_provider_benchmark.md) and [the full-video validation](coreml_full_validation.md).
