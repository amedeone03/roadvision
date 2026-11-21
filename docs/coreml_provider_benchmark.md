# ONNX Runtime Core ML provider benchmark

Experimental check of `CoreMLExecutionProvider` on the existing `yolo11n.onnx`. PyTorch/MPS stays the default. No production algorithm, threshold, or dependency changed. Native Core ML export was not attempted.

The session is ONNX Runtime 1.32.0 in `.venv-onnx` (Python 3.12.16, PyTorch 2.15.0, Ultralytics 8.4.193, NumPy 2.5.4, ONNX 1.23.2). Available providers are `CoreMLExecutionProvider`, `AzureExecutionProvider`, and `CPUExecutionProvider`.

## Configuration

The timed session requests `CoreMLExecutionProvider` and keeps `CPUExecutionProvider` as fallback.

| Option | Value |
| --- | --- |
| ModelFormat | MLProgram |
| MLComputeUnits | ALL |
| RequireStaticInputShapes | 1 |
| EnableOnSubgraphs | 1 |
| SpecializationStrategy | FastPrediction |
| ProfileComputePlan | 0 |
| AllowLowPrecisionAccumulationOnGPU | 0 |

This runtime accepts `MLComputeUnits` values `ALL`, `CPUAndGPU`, `CPUOnly`, and `CPUAndNeuralEngine`, and `ModelFormat` values `MLProgram` and `NeuralNetwork`. The exported input is static `[1, 3, 640, 640]`, so `RequireStaticInputShapes=1` does not reject it. The timed Ultralytics session reported `CoreMLExecutionProvider`, `CPUExecutionProvider`.

## What actually ran

A profiled session with the same options assigned **320 of 320** graph nodes to one Core ML partition. The profile contains **1 Core ML kernel and 0 CPU kernels**. `CPUExecutionProvider` was registered and received no kernels. No unsupported-operator lines were logged. The steady-state times below come from a separate session with profiling turned off.

The NeuralNetwork format, which Ultralytics selects when `device=mps` passes only `COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES`, supported 314 of 320 nodes. Its profile left 2 `MatMul` kernels and 4 `Concat` kernels on CPU, in 5 Core ML partitions. The timed comparison uses MLProgram.

## Latency

One hundred frames. Wall clock around one Ultralytics `predict()` or `track()` call, including preprocessing and Python NMS. PyTorch/MPS synchronizes around the call. ONNX uses ordinary wall-clock time. The first frame is warm-up. These three columns are the same run. The earlier approved CPU experiment measured PyTorch/MPS at 13.450 ms and ONNX CPU at 17.254 ms; those passes are not added to this table.

| | PyTorch/MPS | ONNX CPU | ONNX Core ML |
| --- | ---: | ---: | ---: |
| Model load | 832.894 ms | 6.693 ms | 5.836 ms |
| First `predict()` | 1357.027 ms | 57.228 ms | 270.298 ms |
| Steady `predict()` mean / median / p95 / p99 | 15.417 / 13.744 / 21.210 / 23.041 ms | 18.399 / 17.644 / 21.061 / 29.965 ms | 5.702 / 5.596 / 6.186 / 7.178 ms |
| Steady `predict()` throughput | 64.863 frames/s | 54.351 frames/s | 175.391 frames/s |
| Steady `track()` mean | 16.539 ms | 18.221 ms | 12.349 ms |
| Steady `track()` throughput | 60.462 frames/s | 54.882 frames/s | 80.976 frames/s |

On this Apple M4, Core ML `predict()` is **9.715 ms faster** than synchronized PyTorch/MPS in the same run (5.702 ms versus 15.417 ms), and faster than ONNX CPU at 18.399 ms. The 270 ms first prediction is the compilation warm-up. There is no model cache directory, so a new process compiles again.

## Detections

Greedy one-to-one matching within class at IoU >= 0.5, after Ultralytics NMS and before tracking.

Core ML versus PyTorch/MPS: 315 of 318 PyTorch boxes matched, median IoU 0.998, minimum 0.909, median center shift 0.126 px. The 3 unmatched PyTorch boxes were at or below confidence 0.30. This is the same pattern as the CPU ONNX comparison.

Core ML versus ONNX CPU: 315 of 315 boxes matched, median and minimum IoU 1.000, center shift 0 px, confidence gap 0 on this sample.

## Tracking

Fresh ByteTrack state and the current tracker file. Trajectories are mapped by box overlap.

Core ML versus ONNX CPU: 6 of 6 tracks mapped, all 6 kept the same ID, no fragmentation and no identity-switch frames. The ID differences against PyTorch are the PyTorch-versus-ONNX differences, not an extra Core ML effect.

Core ML versus PyTorch/MPS: 6 of 6 tracks mapped, 2 kept the same ID, 4 were renumbered, 2 reference tracks overlapped more than one Core ML ID, and 6 frames inside the mapped tracks preferred a different ID. No reference track was left unmatched.

## Pipeline

The unchanged pipeline cannot select these provider options. `device=mps` on an ONNX file takes Ultralytics' NeuralNetwork flag path. This spot check forced the MLProgram providers around `run_video` without editing `pipeline.py` or `tracks.py`.

One hundred frames produced 280 observations and 3 episodes in 2.885 s. The PyTorch reference in `runs/onnx_benchmark/pytorch_pipeline` had 278 observations, 3 episodes, and a 3.814 s processing duration. All three episode windows match: frames 38–46, 67–71, and 81–83. The third episode is track 10 on PyTorch and track 8 on Core ML. Strict `(frame, track_id)` keys therefore still disagree. This is not an accuracy claim.

## Recommendation

Core ML through ONNX Runtime is worth a later optional Apple Silicon path. The MLProgram provider ran the whole graph, and steady-state `predict()` was faster than PyTorch/MPS and ONNX CPU on this machine, with episode windows unchanged. It should not become the default yet. The minimal integration change is a way to pass this provider configuration into the existing Ultralytics session. Native Core ML export is a separate experiment and is not the next step; this provider already executes the current ONNX graph.

```bash
.venv-onnx/bin/python scripts/benchmark_coreml.py --max-frames 100
```
